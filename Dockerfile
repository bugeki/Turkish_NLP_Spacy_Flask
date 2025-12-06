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

# Download Turkish spaCy model - using correct working URL
RUN pip install https://huggingface.co/turkish-nlp-suite/tr_core_news_md/resolve/main/tr_core_news_md-1.0-py3-none-any.whl

# Copy application code
COPY . .

# Test imports before running
RUN python test_import.py || true

# Expose port
EXPOSE 5000

# Command to run the application (Railway provides $PORT automatically)
CMD ["sh", "-c", "echo 'Starting app on port $PORT' && gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 --log-level info --access-logfile - --error-logfile - wsgi:app"]