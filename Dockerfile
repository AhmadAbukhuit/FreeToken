# Use CUDA 13.0 devel image as base since kernels are JIT-compiled on first use
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

# Avoid tzdata prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and necessary build tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /workspace/FreeToken

# Copy the FreeToken source code
COPY . /workspace/FreeToken

# Create a virtual environment and install FreeToken with acceleration
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install -e ".[accel]"

# Install Gateway dependencies
RUN . /opt/venv/bin/activate && \
    uv pip install fastapi uvicorn httpx

# Ensure the virtualenv is in PATH so we don't need to source it every time
ENV PATH="/opt/venv/bin:$PATH"

# Expose the API Gateway port
EXPOSE 8080

# Entry point for the Gateway Server
ENTRYPOINT ["uvicorn", "gateway:app"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
