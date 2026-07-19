from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database.connection import get_db
from backend.database.models import PredictionLog
from backend.services.prediction_service import predict_pneumonia
from backend.config import settings
import time
import os
import glob

router = APIRouter()

@router.post("/predict/{file_id}")
async def make_prediction(file_id: str, db: Session = Depends(get_db)):
    start_time = time.time()
    
    search_pattern = os.path.join(settings.UPLOAD_DIR, f"{file_id}_*")
    matches = glob.glob(search_pattern)
    if not matches:
        raise HTTPException(status_code=404, detail="File not found")
    file_path = matches[0]
    
    try:
        result = predict_pneumonia(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    latency = (time.time() - start_time) * 1000
    
    log_entry = PredictionLog(
        image_id=file_id,
        prediction=result["prediction"],
        confidence=result["confidence"],
        model_version=result["model_version"],
        latency_ms=latency
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)
    
    return {"result": result, "latency_ms": latency}
