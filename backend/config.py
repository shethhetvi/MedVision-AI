from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "MedVision AI API"
    PROJECT_VERSION: str = "1.0.0"
    
    # Using SQLite by default for easier local development
    DATABASE_URL: str = "sqlite:///./medvision.db"
    
    UPLOAD_DIR: str = "data/uploads"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
