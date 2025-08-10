#!/usr/bin/env python3
"""
Startup script for Donizo Semantic Pricing Engine

Checks dependencies, sets up database, and starts the server.
Handles environment configuration and basic error checking.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]} detected")

def check_environment_file():
    """Check if environment configuration exists"""
    env_files = ["config.env", ".env", "env.example"]
    
    for env_file in env_files:
        if Path(env_file).exists():
            print(f"✅ Environment file found: {env_file}")
            return env_file
    
    print("⚠️  No environment file found, creating from template...")
    create_default_env()
    return "config.env"

def create_default_env():
    """Create a default environment file"""
    default_config = """# DONIZO SEMANTIC PRICING ENGINE - ENVIRONMENT CONFIGURATION
# ==========================================================

# OpenAI Configuration (REQUIRED - Add your key here)
OPENAI_API_KEY=your-openai-key-here
OPENAI_MODEL=text-embedding-3-small

# Database Configuration
DATABASE_URL=postgresql://localhost/donizo_production
DB_HOST=localhost
DB_PORT=5432
DB_NAME=donizo_production

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false
LOG_LEVEL=INFO

# Application Configuration
APP_NAME=Donizo Semantic Pricing Engine
APP_VERSION=2.0.0
MATERIALS_COUNT_TARGET=3600
RESPONSE_TIME_TARGET_MS=500

# Business Logic Configuration
DEFAULT_MARGIN_RATE=0.25
VAT_RATE_RENOVATION=0.10
VAT_RATE_NEW_BUILD=0.20
DEFAULT_LABOR_RATE_PER_HOUR=35.0

# Multilingual Configuration
SUPPORTED_LANGUAGES=en,fr,es,it
DEFAULT_LANGUAGE=en
"""
    
    with open("config.env", "w") as f:
        f.write(default_config)
    
    print("📝 Created config.env - Please add your OpenAI API key!")

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import psycopg2
        import openai
        import sentence_transformers
        import pydantic_settings
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install with: pip install -r requirements_organized.txt")
        return False

def check_database():
    """Check if PostgreSQL is running and database exists"""
    try:
        import psycopg2
        
        # Try to connect to default postgres database first
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="postgres",
            user="postgres"
        )
        conn.close()
        print("✅ PostgreSQL is running")
        
        # Try to connect to our database
        try:
            conn = psycopg2.connect(
                host="localhost", 
                port=5432,
                database="donizo_production",
                user="postgres"
            )
            conn.close()
            print("✅ Database 'donizo_production' exists")
        except psycopg2.OperationalError:
            print("⚠️  Database 'donizo_production' not found, will create it")
            create_database()
        
        return True
        
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        print("💡 Make sure PostgreSQL is running: brew services start postgresql")
        return False

def create_database():
    """Create the production database"""
    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
        
        conn = psycopg2.connect(
            host="localhost",
            port=5432, 
            database="postgres",
            user="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE donizo_production;")
        cursor.close()
        conn.close()
        
        # Add pgvector extension
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="donizo_production", 
            user="postgres"
        )
        cursor = conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ Database 'donizo_production' created with pgvector")
        
    except Exception as e:
        print(f"❌ Failed to create database: {e}")

def check_openai_key():
    """Check if OpenAI API key is configured"""
    from config import get_settings
    
    try:
        settings = get_settings()
        if settings.has_openai_key():
            # Test the key by making a small request
            import openai
            client = openai.OpenAI(api_key=settings.openai.api_key)
            client.embeddings.create(
                model="text-embedding-3-small",
                input="test"
            )
            print("✅ OpenAI API key is valid")
            return True
        else:
            print("⚠️  No OpenAI API key configured - will use fallback model")
            return True
    except Exception as e:
        if "quota" in str(e).lower():
            print("⚠️  OpenAI quota exceeded - will use fallback model")
            return True
        else:
            print(f"❌ OpenAI API key test failed: {e}")
            return False

def start_application():
    """Start the organized application"""
    print("\n🚀 Starting Donizo Semantic Pricing Engine (Organized Version)...")
    print("=" * 60)
    
    try:
        # Import and run the organized app
        from app_organized import app, settings
        import uvicorn
        
        print(f"🌐 Server starting on http://{settings.server.host}:{settings.server.port}")
        print("🔍 API Documentation: http://localhost:8000/docs")
        print("📊 Configuration: http://localhost:8000/config")
        print("\n💡 Press Ctrl+C to stop the server\n")
        
        uvicorn.run(
            "app_organized:app",
            host=settings.server.host,
            port=settings.server.port,
            log_level=settings.server.log_level.lower(),
            reload=settings.server.debug
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

def main():
    """Main startup routine"""
    print("""
Donizo Semantic Pricing Engine - Startup
=========================================

Initializing semantic pricing engine...
""")
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("Environment Config", check_environment_file), 
        ("Dependencies", check_dependencies),
        ("Database", check_database),
        ("OpenAI API", check_openai_key)
    ]
    
    for check_name, check_func in checks:
        print(f"\n🔍 Checking {check_name}...")
        try:
            result = check_func()
            if result is False:
                print(f"❌ {check_name} check failed")
                sys.exit(1)
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
            sys.exit(1)
    
    print("\n✅ All checks passed!")
    
    # Start the application
    start_application()

if __name__ == "__main__":
    main()
