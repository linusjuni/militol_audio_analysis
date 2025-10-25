from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    OPENAI_API_KEY: str = Field(
        ..., description="OpenAI API key for intelligence generation"
    )

    # LLM Configuration
    LLM_MODEL: str = Field(
        default="gpt-4o-mini", description="OpenAI model for intelligence summaries"
    )
    
    LLM_MAX_TOKENS: int = Field(default=1000, description="Max tokens for LLM response")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
