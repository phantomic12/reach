# Reach — Docker image for web server
# Build: docker build -t reach-server .
# Run:   docker run --gpus all -p 8501:8501 reach-server

FROM python:3.12-slim

WORKDIR /app

# System dependencies for PDF processing and GPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    libqpdf-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
COPY reach-core/ reach-core/
RUN pip install --no-cache-dir -e "reach-core[server,surya]"

# Copy server code
COPY reach-server/ reach-server/

# Create model cache directory (models mounted at runtime or downloaded)
RUN mkdir -p /root/.reach/models

# Expose web UI port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/health')"

# Run the server
CMD ["uvicorn", "reach-server.main:app", "--host", "0.0.0.0", "--port", "8501"]
