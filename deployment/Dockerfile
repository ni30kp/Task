# Donizo Complete Semantic Pricing Engine - Docker Configuration
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker caching
COPY requirements_complete.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_complete.txt

# Copy application code
COPY complete_app.py .
COPY static/ static/
COPY README_COMPLETE.md .

# Create directory for database
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

# Run the application
CMD ["python3", "complete_app.py"]