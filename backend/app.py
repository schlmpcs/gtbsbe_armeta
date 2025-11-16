import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from inference_enhanced_fixed import FastPDFProcessor
from utils import cleanup_old_files, ensure_directories

app = FastAPI(title="PDF Processor API", version="2.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize directories
UPLOAD_DIR = Path("storage/uploads")
OUTPUT_DIR = Path("storage/outputs")
ensure_directories()

# Initialize processor (loads model once)
try:
    processor = FastPDFProcessor("models/best.pt")
    print(f"✅ Processor initialized successfully")
except Exception as e:
    print(f"⚠️ Warning: Failed to initialize processor: {e}")
    print("The /process endpoint will not work until the model is available")
    processor = None

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

@app.on_event("startup")
async def startup_event():
    """Cleanup old files on startup"""
    cleanup_old_files(UPLOAD_DIR)
    cleanup_old_files(OUTPUT_DIR)
    if processor:
        print(f"Server started. Using device: {processor.device}")
        print(f"Detected signature classes: {processor.sig_class_ids}")
        print(f"Detected stamp classes: {processor.stamp_class_ids}")
    else:
        print("⚠️ Server started without model - check models/best.pt")

@app.post("/process")
async def process_pdfs(
    files: List[UploadFile] = File(...),
    save_json: bool = Query(True, description="Save annotations as JSON"),
    return_json: bool = Query(False, description="Return annotations in response")
):
    """
    Process multiple PDFs with enhanced features:
    - QR code detection with clickable links
    - Colored bounding boxes with labels
    - Area filtering for signatures and stamps
    - Optional JSON annotations output
    """
    if processor is None:
        raise HTTPException(
            status_code=503, 
            detail="Processor not initialized. Please ensure models/best.pt exists."
        )
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Save uploaded files
    upload_paths = []
    for file in files:
        if not file.filename.endswith('.pdf'):
            continue
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        upload_paths.append(file_path)
    
    if not upload_paths:
        raise HTTPException(status_code=400, detail="No PDF files found")
    
    print(f"📤 Received {len(upload_paths)} PDF(s) for processing")
    
    # Process PDFs with enhanced processor
    loop = asyncio.get_event_loop()
    try:
        output_paths, annotations = await loop.run_in_executor(
            executor,
            processor.process_batch,
            upload_paths,
            OUTPUT_DIR,
            save_json
        )
    except Exception as e:
        # Cleanup uploaded files on error
        for path in upload_paths:
            path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    if not output_paths:
        # Cleanup uploaded files if no outputs
        for path in upload_paths:
            path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="No PDFs were processed successfully")
    
    # Create ZIP file
    zip_path = OUTPUT_DIR / "processed.zip"
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add processed PDFs
            for output_path in output_paths:
                zipf.write(output_path, output_path.name)
            
            # Add JSON if it exists
            json_path = OUTPUT_DIR / "annotations.json"
            if json_path.exists():
                zipf.write(json_path, "annotations.json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create ZIP: {str(e)}")
    
    # Cleanup uploaded files
    for path in upload_paths:
        path.unlink(missing_ok=True)
    
    print(f"✅ Successfully processed {len(output_paths)} PDF(s)")
    
    # Prepare response
    response_data = {
        "message": f"Successfully processed {len(output_paths)} PDFs",
        "processed_files": [p.name for p in output_paths],
        "download_url": "/download/processed.zip"
    }
    
    if return_json and annotations:
        response_data["annotations"] = annotations
    
    # Return appropriate response
    if return_json:
        return JSONResponse(content=response_data)
    else:
        return FileResponse(
            path=zip_path,
            media_type='application/zip',
            filename="processed_pdfs.zip"
        )

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a processed file"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        media_type='application/octet-stream',
        filename=filename
    )

@app.get("/health")
async def health_check():
    """Health check endpoint with system info"""
    if processor is None:
        return {
            "status": "degraded",
            "error": "Processor not initialized",
            "device": "unknown",
            "model_loaded": False
        }
    
    return {
        "status": "healthy",
        "device": processor.device,
        "model_loaded": True,
        "signature_classes": processor.sig_class_ids,
        "stamp_classes": processor.stamp_class_ids
    }

@app.get("/stats")
async def get_stats():
    """Get processing statistics"""
    upload_files = list(UPLOAD_DIR.glob("*.pdf"))
    output_files = list(OUTPUT_DIR.glob("*.pdf"))
    
    return {
        "pending_uploads": len(upload_files),
        "processed_files": len(output_files),
        "storage": {
            "uploads_mb": sum(f.stat().st_size for f in upload_files) / (1024 * 1024),
            "outputs_mb": sum(f.stat().st_size for f in output_files) / (1024 * 1024)
        }
    }

@app.delete("/cleanup")
async def cleanup_storage():
    """Manual cleanup of old files"""
    cleanup_old_files(UPLOAD_DIR, hours=0)
    cleanup_old_files(OUTPUT_DIR, hours=0)
    return {"message": "Storage cleaned up successfully"}

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "PDF Processor API",
        "version": "2.0.0",
        "endpoints": {
            "POST /process": "Process PDFs with QR, signature, and stamp detection",
            "GET /download/{filename}": "Download processed files",
            "GET /health": "Health check",
            "GET /stats": "Processing statistics",
            "DELETE /cleanup": "Cleanup old files"
        },
        "model_status": "loaded" if processor else "not loaded"
    }

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )