from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "MedVision AI API"
    PROJECT_VERSION: str = "1.0.0"
    
    # Using SQLite by default for easier local development
    DATABASE_URL: str = "sqlite:///./medvision.db"
    
    UPLOAD_DIR: str = "data/uploads"

    class Config:
        env_file = ".env"

settings = Settings()
