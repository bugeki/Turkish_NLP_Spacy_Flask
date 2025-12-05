FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (without Turkish model yet)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download Turkish spaCy model using spacy download
RUN python -m spacy download tr_core_news_md

# Copy application code
COPY . .

# Test imports before running
RUN python test_import.py || true

# Expose port
EXPOSE 5000

# Command to run the application (Railway provides $PORT automatically)
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 2 --timeout 120 --log-level info wsgi:app