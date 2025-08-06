# Multi-stage build for smaller production image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set labels for better image management
LABEL maintainer="sadman.sadat@vivasoftltd.com"
LABEL version="1.0"
LABEL description="Bengali Gemma3 Inference Server"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Create non-root user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser app.py .

# Switch to non-root user
USER appuser

# Update PATH to include local packages
ENV PATH=/root/.local/bin:$PATH

# Set Python path
ENV PYTHONPATH=/root/.local/lib/python3.10/site-packages:$PYTHONPATH

# Expose port
EXPOSE 8000

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Run the application
CMD ["python", "app.py"]