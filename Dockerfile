FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/app/ ./

# Default port
ENV PORT=8000
EXPOSE 8000

# Start command - use shell form for variable expansion
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
