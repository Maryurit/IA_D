FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema (OpenCV, etc.)
RUN apt-get update && apt-get install -y \
    libgl1 \            
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

RUN apt-get update && apt-get install -y curl

# Healthcheck opcional
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5001/health || exit 1

CMD ["python", "streaming_server.py"]