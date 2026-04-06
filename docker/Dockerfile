FROM python:3.11-slim

LABEL maintainer="Genome Sequencing & Reconstruction"
LABEL description="Advanced ancient DNA reconstruction with DNABERT-2, ESM, AlphaFold attention"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install MAFFT
RUN wget -q https://mafft.cbrc.jp/alignment/software/mafft_7.520-1_amd64.deb \
    && dpkg -i mafft_7.520-1_amd64.deb \
    && rm mafft_7.520-1_amd64.deb \
    || echo "MAFFT install failed — will use Python fallback"

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create directories
RUN mkdir -p data/sequences data/simulated data/alignments data/mappings \
    models/checkpoints results/visualizations

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default: start API server
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
