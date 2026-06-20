FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_USE_LEGACY_KERAS=1 \
    MAX_SAMPLES=-1

WORKDIR /workspace

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pre-install dependencies to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY setup.py .
COPY README.md .
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY models/distilbert_imdb_tf_model/ ./models/distilbert_imdb_tf_model/


# Install application package in editable mode
RUN pip install --no-cache-dir -e .

# Expose FastAPI port
EXPOSE 7860

# Start FastAPI using uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]

