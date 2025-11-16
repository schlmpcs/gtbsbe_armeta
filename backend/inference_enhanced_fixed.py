import os
import io
import json
import torch
import cv2
import numpy as np
import zxingcpp
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from PIL import Image
import fitz  # PyMuPDF
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing as mp

# Environment fix for OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------
# CONFIGURATION
# -----------------------
DPI = 300

# Per-category YOLO settings
SIG_IMGSZ = 1280
SIG_CONF = 0.08
SIG_IOU = 0.35
SIG_MAX_DET = 100

STAMP_IMGSZ = 1280
STAMP_CONF = 0.25
STAMP_IOU = 0.35
STAMP_MAX_DET = 100

# Area filters (in image pixels)
SIGNATURE_MIN_AREA = 1000
SIGNATURE_MAX_AREA = 4000000

STAMP_MIN_AREA = 1000
STAMP_MAX_AREA = 1500000

# Colors (R,G,B) in 0-1 range for PyMuPDF
COLOR_QR = (1, 0, 0)         # red
COLOR_SIGNATURE = (0, 0, 1)  # blue
COLOR_STAMP = (0, 0.4, 0)    # green
COLOR_OTHER = (1, 0, 0)      # default red

WHITE = (1, 1, 1)
LABEL_HEIGHT = 16  # height of top banner in PDF units

# Global lock for thread-safe YOLO inference
YOLO_LOCK = Lock()


def normalize_category(cls_name: str) -> str:
    """Map YOLO class name to normalized category."""
    name = cls_name.lower()
    
    if any(s in name for s in ["signature", "sign"]):
        return "signature"
    if "seal" in name or "stamp" in name:
        return "stamp"
    return name


def get_color_for_category(category: str) -> Tuple[float, float, float]:
    """Get color for a specific category."""
    cat = category.lower()
    if cat == "qr":
        return COLOR_QR
    if cat == "signature":
        return COLOR_SIGNATURE
    if cat == "stamp":
        return COLOR_STAMP
    return COLOR_OTHER


def looks_like_url(text: str) -> bool:
    """Check if text looks like a URL."""
    if not text:
        return False
    t = text.strip().lower()
    return t.startswith("http://") or t.startswith("https://")


def draw_box_with_label(page, rect, label_text: str, color: Tuple[float, float, float]):
    """
    Draws:
    - colored bbox (stroke)
    - colored banner at top of bbox
    - white label text inside banner
    """
    # Draw bbox
    page.draw_rect(rect, color=color, width=1, overlay=True)
    
    # Label banner
    x1, y1, x2, y2 = rect
    top = max(y1 - LABEL_HEIGHT, 0)
    label_rect = fitz.Rect(x1, top, x2, y1)
    
    # Filled banner
    page.draw_rect(label_rect, color=color, fill=color, width=0, overlay=True)
    
    # Text position
    text_x = x1 + 2
    text_y = top + LABEL_HEIGHT - 2
    page.insert_text(
        fitz.Point(text_x, text_y),
        label_text,
        fontsize=14,
        color=WHITE,
        overlay=True,
    )


def get_class_ids_by_keywords(yolo_names, keywords: List[str]) -> List[int]:
    """Build a list of class IDs whose names contain any of the given keywords."""
    ids = []
    if isinstance(yolo_names, dict):
        items = yolo_names.items()
    else:
        items = enumerate(yolo_names)
    
    kw_lower = [k.lower() for k in keywords]
    
    for idx, name in items:
        n = str(name).lower()
        if any(k in n for k in kw_lower):
            ids.append(idx)
    return ids


class FastPDFProcessor:
    """
    Optimized PDF processor with:
    - QR code detection with clickable links
    - Colored bounding boxes with labels
    - Area filtering for signatures and stamps
    - Batch GPU inference for speed
    - Optional JSON annotations output
    """
    
    def __init__(self, model_path: str):
        """Initialize processor with YOLO model"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing FastPDFProcessor on device: {self.device}")
        
        # Load YOLO model
        self.model = YOLO(model_path)
        if self.device == 'cuda':
            self.model.to('cuda')
        
        # Get class names and identify signature/stamp classes
        self.yolo_names = self.model.names
        self.sig_class_ids = get_class_ids_by_keywords(
            self.yolo_names, ["signature", "sign"]
        )
        self.stamp_class_ids = get_class_ids_by_keywords(
            self.yolo_names, ["stamp", "seal"]
        )
        
        print(f"Model loaded - Signature classes: {self.sig_class_ids}, Stamp classes: {self.stamp_class_ids}")
        
        self.cpu_count = mp.cpu_count()
    
    def process_batch(self, pdf_paths: List[Path], output_dir: Path, 
                     save_json: bool = True) -> Tuple[List[Path], Optional[Dict]]:
        """
        Process multiple PDFs with parallel processing.
        
        Args:
            pdf_paths: List of PDF file paths to process
            output_dir: Directory to save processed PDFs
            save_json: Whether to save annotations as JSON
            
        Returns:
            Tuple of (output_paths, annotations_dict)
        """
        output_paths = []
        all_annotations = {}
        
        # Process PDFs in parallel using threads
        max_workers = min(self.cpu_count, 4)
        print(f"Processing {len(pdf_paths)} PDFs with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_single_pdf, pdf_path, output_dir): pdf_path
                for pdf_path in pdf_paths
            }
            
            for future in as_completed(futures):
                pdf_path = futures[future]
                try:
                    output_path, annotations = future.result()
                    if output_path:
                        output_paths.append(output_path)
                        if save_json and annotations:
                            all_annotations[pdf_path.name] = annotations
                except Exception as e:
                    print(f"Error processing {pdf_path}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Save combined JSON if requested
        if save_json and all_annotations:
            json_path = output_dir / "annotations.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_annotations, f, ensure_ascii=False, indent=2)
            print(f"Annotations saved to: {json_path}")
        
        return output_paths, all_annotations if save_json else None
    
    def _process_single_pdf(self, pdf_path: Path, output_dir: Path) -> Tuple[Optional[Path], Dict]:
        """
        Process a single PDF with QR detection, YOLO inference, and annotations.
        
        Returns:
            Tuple of (output_path, annotations_dict)
        """
        try:
            doc = fitz.open(str(pdf_path))
            pdf_name = pdf_path.name
            annotations = {}
            annotation_id = 1
            
            print(f"[{pdf_name}] Processing {len(doc)} pages - sig_ids={self.sig_class_ids}, stamp_ids={self.stamp_class_ids}")
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_key = f"page_{page_num + 1}"
                
                # Render page to image
                zoom = DPI / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Convert to numpy array
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                
                # Convert to BGR for OpenCV
                if pix.n == 4:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif pix.n == 1:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    img_bgr = img
                
                page_width = pix.width
                page_height = pix.height
                
                page_entry = {
                    "page_number": page_num + 1,
                    "width": page_width,
                    "height": page_height,
                    "annotations": []
                }
                
                # ========== 1. QR Code Detection ==========
                qr_results = zxingcpp.read_barcodes(img_bgr)
                
                for qr in qr_results:
                    if "QR" not in str(qr.format):
                        continue
                    
                    # Get bounding box from position
                    pos = qr.position
                    pts = [
                        (pos.top_left.x, pos.top_left.y),
                        (pos.top_right.x, pos.top_right.y),
                        (pos.bottom_right.x, pos.bottom_right.y),
                        (pos.bottom_left.x, pos.bottom_left.y),
                    ]
                    
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    
                    x_min, y_min = float(min(xs)), float(min(ys))
                    x_max, y_max = float(max(xs)), float(max(ys))
                    
                    # Image space bbox
                    w_img = x_max - x_min
                    h_img = y_max - y_min
                    area = w_img * h_img
                    
                    # Convert to PDF coordinates
                    rect = fitz.Rect(x_min/zoom, y_min/zoom, x_max/zoom, y_max/zoom)
                    
                    # Draw QR box with label
                    qr_text = qr.text
                    label_text = f"QR: {qr_text[:20]}..." if len(qr_text) > 20 else f"QR: {qr_text}"
                    draw_box_with_label(page, rect, label_text, COLOR_QR)
                    
                    # Add clickable link if URL
                    if looks_like_url(qr_text):
                        page.insert_link({
                            'kind': fitz.LINK_URI,
                            'from': rect,
                            'uri': qr_text
                        })
                    
                    # Store annotation
                    page_entry["annotations"].append({
                        f"annotation_{annotation_id}": {
                            "category": "qr",
                            "bbox": {
                                "x": x_min,
                                "y": y_min,
                                "width": w_img,
                                "height": h_img
                            },
                            "area": area,
                            "text": qr_text,
                            "label": label_text
                        }
                    })
                    annotation_id += 1
                
                # ========== 2. Signature Detection ==========
                if self.sig_class_ids:
                    with YOLO_LOCK:
                        sig_results = self.model(
                            img_bgr,
                            verbose=False,
                            imgsz=SIG_IMGSZ,
                            conf=SIG_CONF,
                            iou=SIG_IOU,
                            max_det=SIG_MAX_DET,
                            classes=self.sig_class_ids
                        )
                    
                    yres_sig = sig_results[0]
                    if yres_sig.boxes is not None and len(yres_sig.boxes) > 0:
                        xyxy = yres_sig.boxes.xyxy.cpu().numpy()
                        cls_ids = yres_sig.boxes.cls.cpu().numpy().astype(int)
                        confs = yres_sig.boxes.conf.cpu().numpy()
                        
                        for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, cls_ids, confs):
                            # Calculate area and apply filter
                            w_img = float(x2 - x1)
                            h_img = float(y2 - y1)
                            area = w_img * h_img
                            
                            if area < SIGNATURE_MIN_AREA or area > SIGNATURE_MAX_AREA:
                                continue
                            
                            # Get category name
                            if isinstance(self.yolo_names, dict):
                                cls_name = self.yolo_names.get(cls_id, str(cls_id))
                            else:
                                cls_name = str(self.yolo_names[cls_id])
                            category = normalize_category(cls_name)
                            
                            # Convert to PDF coordinates
                            rect = fitz.Rect(x1/zoom, y1/zoom, x2/zoom, y2/zoom)
                            
                            # Draw box with label
                            color = get_color_for_category(category)
                            label_text = f"{category} {conf:.2f}"
                            draw_box_with_label(page, rect, label_text, color)
                            
                            # Store annotation
                            page_entry["annotations"].append({
                                f"annotation_{annotation_id}": {
                                    "category": category,
                                    "bbox": {
                                        "x": float(x1),
                                        "y": float(y1),
                                        "width": w_img,
                                        "height": h_img
                                    },
                                    "area": area,
                                    "confidence": float(conf),
                                    "label": label_text
                                }
                            })
                            annotation_id += 1
                
                # ========== 3. Stamp/Seal Detection ==========
                if self.stamp_class_ids:
                    with YOLO_LOCK:
                        stamp_results = self.model(
                            img_bgr,
                            verbose=False,
                            imgsz=STAMP_IMGSZ,
                            conf=STAMP_CONF,
                            iou=STAMP_IOU,
                            max_det=STAMP_MAX_DET,
                            classes=self.stamp_class_ids
                        )
                    
                    yres_stamp = stamp_results[0]
                    if yres_stamp.boxes is not None and len(yres_stamp.boxes) > 0:
                        xyxy = yres_stamp.boxes.xyxy.cpu().numpy()
                        cls_ids = yres_stamp.boxes.cls.cpu().numpy().astype(int)
                        confs = yres_stamp.boxes.conf.cpu().numpy()
                        
                        for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, cls_ids, confs):
                            # Calculate area
                            w_img = float(x2 - x1)
                            h_img = float(y2 - y1)
                            area = w_img * h_img
                            
                            # Get category name
                            if isinstance(self.yolo_names, dict):
                                cls_name = self.yolo_names.get(cls_id, str(cls_id))
                            else:
                                cls_name = str(self.yolo_names[cls_id])
                            category = normalize_category(cls_name)
                            
                            # Apply area filter for stamps
                            if category == "stamp":
                                if area < STAMP_MIN_AREA or area > STAMP_MAX_AREA:
                                    continue
                            
                            # Convert to PDF coordinates
                            rect = fitz.Rect(x1/zoom, y1/zoom, x2/zoom, y2/zoom)
                            
                            # Draw box with label
                            color = get_color_for_category(category)
                            label_text = f"{category} {conf:.2f}"
                            draw_box_with_label(page, rect, label_text, color)
                            
                            # Store annotation
                            page_entry["annotations"].append({
                                f"annotation_{annotation_id}": {
                                    "category": category,
                                    "bbox": {
                                        "x": float(x1),
                                        "y": float(y1),
                                        "width": w_img,
                                        "height": h_img
                                    },
                                    "area": area,
                                    "confidence": float(conf),
                                    "label": label_text
                                }
                            })
                            annotation_id += 1
                
                # Store page annotations
                annotations[page_key] = page_entry
            
            # Save processed PDF
            output_path = output_dir / f"processed_{pdf_name}"
            doc.save(str(output_path))
            doc.close()
            
            print(f"Finished {pdf_name}")
            return output_path, annotations
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            import traceback
            traceback.print_exc()
            return None, {}
