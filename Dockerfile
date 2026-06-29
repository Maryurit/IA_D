FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --retries 10 --timeout 200 torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --retries 10 --timeout 200 --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5001
CMD ["python", "streaming_server.py"]