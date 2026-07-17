from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database.connection import get_db
from backend.database.models import PredictionLog
from backend.services.prediction_service import predict_pneumonia
import time

router = APIRouter()

@router.post("/predict/{file_id}")
async def make_prediction(file_id: str, db: Session = Depends(get_db)):
    start_time = time.time()
    
    # In a real scenario, fetch file from DB or filesystem using file_id
    file_path = f"data/uploads/{file_id}"
    
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
