FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Use a closer mirror and optimize apt-get
RUN echo 'Acquire::http::Timeout "10";' > /etc/apt/apt.conf.d/99timeout && \
    echo 'Acquire::http::Pipeline-Depth "0";' >> /etc/apt/apt.conf.d/99timeout && \
    echo 'Acquire::http::No-Cache "true";' >> /etc/apt/apt.conf.d/99timeout && \
    echo 'Acquire::BrokenProxy "true";' >> /etc/apt/apt.conf.d/99timeout

# Install only essential system dependencies in one layer
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set environment variable
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Install Python packages with better caching
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p storage/uploads storage/outputs models

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]
