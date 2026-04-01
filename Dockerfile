FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for mysqlclient/pymysql compatibility and clean HTTPS calls.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install \
    --no-cache-dir \
    --prefer-binary \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /app/backend/requirements.txt

COPY . /app

WORKDIR /app/backend
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
