from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    UPLOAD_DIR: str = "temp"
    
    GEMINI_API_KEY: str
    LLM_MODEL: str
    EMBEDDING_MODEL: str
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()