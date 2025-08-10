#!/usr/bin/env python3
"""
Configuration management for Donizo Semantic Pricing Engine
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / "config.env"
if env_path.exists():
    load_dotenv(env_path)

class DatabaseConfig(BaseSettings):
    """Database configuration settings"""
    url: str = Field(default="postgresql://localhost/donizo_production", env="DATABASE_URL")
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    name: str = Field(default="donizo_production", env="DB_NAME")
    user: str = Field(default="postgres", env="DB_USER")
    password: str = Field(default="", env="DB_PASSWORD")
    
    class Config:
        env_prefix = "DB_"

class OpenAIConfig(BaseSettings):
    """OpenAI API configuration"""
    api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    model: str = Field(default="text-embedding-3-small", env="OPENAI_MODEL")
    max_retries: int = Field(default=3, env="OPENAI_MAX_RETRIES")
    timeout: int = Field(default=30, env="OPENAI_TIMEOUT")
    
    class Config:
        env_prefix = "OPENAI_"

class ServerConfig(BaseSettings):
    """Server configuration"""
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    class Config:
        env_prefix = "SERVER_"

class ApplicationConfig(BaseSettings):
    """Application-specific configuration"""
    name: str = Field(default="Donizo Semantic Pricing Engine", env="APP_NAME")
    version: str = Field(default="2.0.0", env="APP_VERSION")
    materials_target: int = Field(default=3600, env="MATERIALS_COUNT_TARGET")
    response_time_target: int = Field(default=500, env="RESPONSE_TIME_TARGET_MS")
    
    class Config:
        env_prefix = "APP_"

class VectorConfig(BaseSettings):
    """Vector database configuration"""
    openai_dimension: int = Field(default=1536, env="VECTOR_DIMENSION_OPENAI")
    fallback_dimension: int = Field(default=384, env="VECTOR_DIMENSION_FALLBACK")
    hnsw_m: int = Field(default=16, env="HNSW_M")
    hnsw_ef_construction: int = Field(default=64, env="HNSW_EF_CONSTRUCTION")
    
    class Config:
        env_prefix = "VECTOR_"

class BusinessConfig(BaseSettings):
    """Business logic configuration"""
    margin_rate: float = Field(default=0.25, env="DEFAULT_MARGIN_RATE")
    vat_renovation: float = Field(default=0.10, env="VAT_RATE_RENOVATION")
    vat_new_build: float = Field(default=0.20, env="VAT_RATE_NEW_BUILD")
    labor_rate_per_hour: float = Field(default=35.0, env="DEFAULT_LABOR_RATE_PER_HOUR")
    
    class Config:
        env_prefix = "BUSINESS_"

class FallbackConfig(BaseSettings):
    """Fallback configuration"""
    model: str = Field(default="all-MiniLM-L6-v2", env="FALLBACK_MODEL")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    timeout: int = Field(default=30, env="TIMEOUT_SECONDS")
    
    class Config:
        env_prefix = "FALLBACK_"

class MultilingualConfig(BaseSettings):
    """Multilingual support configuration"""
    supported_languages: List[str] = Field(default=["en", "fr", "es", "it"], env="SUPPORTED_LANGUAGES")
    default_language: str = Field(default="en", env="DEFAULT_LANGUAGE")
    
    class Config:
        env_prefix = "MULTILINGUAL_"
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if isinstance(self.supported_languages, str):
            self.supported_languages = [lang.strip() for lang in self.supported_languages.split(",")]

class PerformanceConfig(BaseSettings):
    """Performance and caching configuration"""
    max_concurrent_requests: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    batch_size: int = Field(default=50, env="BATCH_SIZE")
    
    class Config:
        env_prefix = "PERFORMANCE_"

class Settings:
    """Main configuration class that aggregates all settings"""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.openai = OpenAIConfig()
        self.server = ServerConfig()
        self.app = ApplicationConfig()
        self.vector = VectorConfig()
        self.business = BusinessConfig()
        self.fallback = FallbackConfig()
        self.multilingual = MultilingualConfig()
        self.performance = PerformanceConfig()
    
    def get_database_url(self) -> str:
        """Get the database URL for connections"""
        if self.database.url and self.database.url != "postgresql://localhost/donizo_production":
            return self.database.url
        
        # Build URL from components
        password_part = f":{self.database.password}" if self.database.password else ""
        return f"postgresql://{self.database.user}{password_part}@{self.database.host}:{self.database.port}/{self.database.name}"
    
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is configured"""
        return bool(self.openai.api_key and self.openai.api_key.strip())
    
    def get_vector_dimension(self) -> int:
        """Get the appropriate vector dimension based on available models"""
        return self.vector.openai_dimension if self.has_openai_key() else self.vector.fallback_dimension
    
    def print_config_summary(self):
        """Print a summary of the current configuration"""
        print(f"""
DONIZO CONFIGURATION SUMMARY
==============================

Application:
   • Name: {self.app.name}
   • Version: {self.app.version}
   • Target Materials: {self.app.materials_target:,}
   • Response Time Target: {self.app.response_time_target}ms

Database:
   • URL: {self.get_database_url()}
   • Vector Dimension: {self.get_vector_dimension()}D

AI Models:
   • OpenAI Available: {'YES' if self.has_openai_key() else 'NO'}
   • Primary Model: {self.openai.model if self.has_openai_key() else 'N/A'}
   • Fallback Model: {self.fallback.model}

Multilingual:
   • Supported: {', '.join(self.multilingual.supported_languages)}
   • Default: {self.multilingual.default_language}

Business Logic:
   • Margin Rate: {self.business.margin_rate*100:.1f}%
   • VAT (Renovation): {self.business.vat_renovation*100:.1f}%
   • VAT (New Build): {self.business.vat_new_build*100:.1f}%
   • Labor Rate: €{self.business.labor_rate_per_hour}/hour

Server:
   • Host: {self.server.host}
   • Port: {self.server.port}
   • Debug: {self.server.debug}
   • Log Level: {self.server.log_level}
        """)

# Global settings instance
settings = Settings()

# Convenience functions
def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings

def get_database_url() -> str:
    """Get the database connection URL"""
    return settings.get_database_url()

def get_openai_key() -> Optional[str]:
    """Get the OpenAI API key if available"""
    return settings.openai.api_key if settings.has_openai_key() else None

def get_vector_dimension() -> int:
    """Get the appropriate vector dimension"""
    return settings.get_vector_dimension()

if __name__ == "__main__":
    # Print configuration when run directly
    settings.print_config_summary()