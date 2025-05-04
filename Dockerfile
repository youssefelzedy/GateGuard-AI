FROM python:3.11-slim

WORKDIR /app

COPY . /app

# Install required system libs
RUN apt update && apt install -y libgl1 libglib2.0-0

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --retries 10 --timeout 100 --no-cache-dir -r requirements.txt

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--workers", "1", "--timeout-keep-alive", "60", "--timeout-grace", "60"]
