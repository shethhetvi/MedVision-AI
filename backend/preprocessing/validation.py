import os
from fastapi import UploadFile, HTTPException

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg"}

async def validate_image(file: UploadFile):
    # Check mime type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are allowed.")
        
    # Check file size
    file_size = getattr(file, 'size', None)
    if file_size is None:
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 5MB.")
