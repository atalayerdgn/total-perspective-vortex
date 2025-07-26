# EEG/BCI processing application
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=app:app app/ ./app/

# Create necessary directories with proper permissions
RUN mkdir -p /app/data/epochs/train /app/data/epochs/test /app/models && \
    chown -R app:app /app

# Switch to app user
USER app

# Set Python path
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Expose port for potential web interface (if needed)
EXPOSE 8000

# Default command - can be overridden
CMD ["python", "app/mybci.py"]
