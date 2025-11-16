# PDF Stamp, QR & Signature Detection System, team GTBSBE

## What It Does
This web application automatically detects and marks stamps and signatures in PDF documents using advanced AI technology. Simply upload your PDFs, and the system will process them and return the documents with red bounding boxes around detected stamps and signatures.

## How to Use

## Step 1: Access the Application
Open your web browser and navigate to the provided URL (will be shared separately).

## Step 2: Upload Your PDFs
- **Drag and Drop**: Simply drag your PDF files from your computer and drop them onto the upload area
- **Click to Select**: Click the upload area to browse and select files from your computer
- **Multiple Files**: You can upload multiple PDFs at once for batch processing

## Step 3: Process the Documents
1. After selecting your files, you'll see a list of uploaded PDFs
2. Click the "Process PDFs" button
3. Wait while the AI analyzes your documents (a spinner will show progress)

## Step 4: Download Results
- Once processing is complete, a ZIP file will automatically download
- The ZIP contains all your processed PDFs with stamps and signatures marked in red boxes
- Original PDFs are not modified - you receive new versions with markings

##Repository content
- Ipynb file for training our model
- Ipynb file for inference
- Web app backend and frontend
  
## Features
- ✅ Batch processing of multiple PDFs simultaneously
- ✅ Fast GPU-accelerated detection
- ✅ High accuracy stamp and signature detection
- ✅ Simple drag-and-drop interface
- ✅ Automatic ZIP packaging for multiple files
- ✅ No file size limits (within reasonable bounds)

## What Gets Detected
The system identifies:
- Official stamps and seals
- Handwritten signatures
- Digital signature marks
- Notary stamps
- Company seals
<img width="1116" height="675" alt="image" src="https://github.com/user-attachments/assets/e5301997-a31e-48b4-9194-c537038b8a4a" />

## Output Format
- Processed PDFs maintain original quality
- Red rectangular boxes mark detected items
- All pages are processed automatically
- Files are named as "processed_[original_name].pdf"
<img width="920" height="570" alt="image" src="https://github.com/user-attachments/assets/9b4f2b63-22d6-4381-8545-acaf17cb0a39" />

## Tips for Best Results
1. Use clear, scanned PDFs for best detection accuracy
2. Ensure stamps and signatures are visible and not too faint
3. PDFs with higher resolution typically yield better results

## Privacy & Security
- Files are processed on secure servers
- Uploaded files are automatically deleted after processing
- No data is stored permanently
- Each session is isolated for security

## Browser Compatibility
Works best with:
- Chrome (recommended)
- Firefox
- Safari
- Edge

## Support
If you encounter any issues or the system is not responding, please notify the technical administrator.
