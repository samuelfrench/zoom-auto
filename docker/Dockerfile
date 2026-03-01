# ---- Stage 1: Frontend build ----
FROM node:20-slim AS frontend-builder

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --ignore-scripts
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Runtime ----
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# System dependencies: Python 3.11 (deadsnakes PPA), audio, X11/dbus for Zoom SDK
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gpg-agent \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    # Audio
    pulseaudio \
    ffmpeg \
    libasound2 \
    # X11 / display libs required by Zoom SDK
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libxrender1 \
    libxtst6 \
    # Browser / GTK libs required by Zoom SDK
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libgbm1 \
    libpango-1.0-0 \
    # Core libs
    libglib2.0-0 \
    libdbus-1-3 \
    # Utilities
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create non-root user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy dependency manifest first for layer caching
COPY pyproject.toml uv.lock ./

# Install Python dependencies (as root so uv can write to its cache)
# First, install only third-party deps (no project itself) for Docker layer caching
RUN uv sync --no-install-project --no-dev --frozen

# Copy application source
COPY src/ src/

# Now install the project itself
RUN uv sync --no-dev --frozen

# Copy default config
COPY config/ config/

# Copy pre-built frontend from stage 1
COPY --from=frontend-builder /build/frontend/dist/ frontend/dist/

# Create data directory (will be volume-mounted at runtime)
RUN mkdir -p data && chown -R appuser:appuser /app

# Create cache directory for non-root user
RUN mkdir -p /home/appuser/.cache/uv && chown -R appuser:appuser /home/appuser

# Switch to non-root user
USER appuser

# Re-point uv cache for non-root user
ENV UV_CACHE_DIR="/home/appuser/.cache/uv"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["uv", "run", "python", "-m", "zoom_auto"]
