FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .
COPY start.sh .

# Make start script executable
RUN chmod +x start.sh

# Expose port
EXPOSE 7860

# Set environment variables
ENV API_KEY=sk_live_abc123xyz789_secure_key_2024
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["./start.sh"]
