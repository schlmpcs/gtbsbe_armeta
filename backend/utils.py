import os
import time
from pathlib import Path
from datetime import datetime, timedelta

def cleanup_old_files(directory: Path, hours: int = 1):
    """Remove files older than specified hours"""
    current_time = time.time()
    for file_path in directory.glob("*"):
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > hours * 3600:
                try:
                    file_path.unlink()
                except:
                    pass

def ensure_directories():
    """Ensure required directories exist"""
    dirs = [
        "storage/uploads",
        "storage/outputs",
        "models"
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
