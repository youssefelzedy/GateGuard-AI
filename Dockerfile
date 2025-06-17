FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt update && apt install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    && apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

# Step 1: Install base requirements
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Step 2: Install filterpy from GitHub
RUN pip install --no-cache-dir git+https://github.com/rlabbe/filterpy.git

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "3000", "--reload"]
