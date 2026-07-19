from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from backend.database.connection import get_db
from backend.database.models import PredictionLog
from backend.reports.pdf_generator import generate_pdf_report
from backend.config import settings
import os
import glob

router = APIRouter()

@router.get("/report/{file_id}")
async def get_report(file_id: str, db: Session = Depends(get_db)):
    # Fetch prediction log
    log_entry = db.query(PredictionLog).filter(PredictionLog.image_id == file_id).first()
    
    if not log_entry:
        raise HTTPException(status_code=404, detail="Prediction not found for this file ID")
        
    # Find original file
    search_pattern = os.path.join(settings.UPLOAD_DIR, f"{file_id}_*")
    matches = glob.glob(search_pattern)
    if not matches:
        raise HTTPException(status_code=404, detail="Original image file not found")
    
    original_image_path = matches[0]
    
    # Check for heatmap (assuming it's saved with _cam suffix in same dir)
    base_name, ext = os.path.splitext(original_image_path)
    heatmap_path = f"{base_name}_cam{ext}"
    if not os.path.exists(heatmap_path):
        heatmap_path = None
        
    patient_info = {
        "id": file_id,
        "name": "Anonymous Patient" # Placeholder, could be added to DB later
    }
    
    prediction_result = {
        "prediction": log_entry.prediction,
        "confidence": log_entry.confidence,
        "model_version": log_entry.model_version
    }
    
    try:
        pdf_path = generate_pdf_report(patient_info, prediction_result, original_image_path, heatmap_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")
        
    return FileResponse(
        pdf_path, 
        media_type='application/pdf', 
        filename=f"MedVision_Report_{file_id}.pdf"
    )
