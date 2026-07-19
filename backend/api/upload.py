from fastapi import APIRouter, UploadFile, File, HTTPException
import uuid
import os
from backend.config import settings
from backend.preprocessing.validation import validate_image

router = APIRouter()

UPLOAD_DIR = settings.UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    await validate_image(file)
    
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    return {"file_id": file_id, "file_path": file_path, "filename": file.filename}
