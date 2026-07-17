from sqlalchemy import Column, Integer, String, Float, DateTime
from backend.database.connection import Base
from datetime import datetime

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(String, index=True)
    prediction = Column(String)
    confidence = Column(Float)
    model_version = Column(String)
    latency_ms = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
