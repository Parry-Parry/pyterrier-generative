# Lightweight Dockerfile for pyterrier-generative testing

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install git (needed for pyterrier_rag installation)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install --no-cache-dir uv

# Copy requirements files
COPY requirements-dev.txt ./
COPY requirements.txt ./

# Install test dependencies
RUN uv pip install --system -r requirements.txt
RUN uv pip install --system -r requirements-dev.txt

# Copy project files
COPY pyterrier_generative ./pyterrier_generative
COPY tests ./tests
COPY pytest.ini ./
COPY pyproject.toml ./

# Install package without dependencies
RUN uv pip install --system --no-deps -e .

# Run tests by default
CMD ["pytest", "-v", "--cov=pyterrier_generative", "--cov-report=term-missing"]
