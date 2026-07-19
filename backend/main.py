from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.config import settings
from backend.api import upload, predict
from backend.database.connection import engine, Base
from backend.database import models
from backend.services.prediction_service import init_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create database tables
    models.Base.metadata.create_all(bind=engine)
    # Initialize the ML model on startup
    init_model()
    yield
    # Clean up on shutdown if needed

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(predict.router, prefix="/api", tags=["Predict"])

@app.get("/")
def read_root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}"}
