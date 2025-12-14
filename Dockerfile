# Dockerfile for pyterrier-generative testing

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY requirements.txt requirements-dev.txt pyproject.toml ./
COPY pyterrier_generative ./pyterrier_generative
COPY tests ./tests
COPY pytest.ini ./

# Install dependencies
RUN uv pip install --system -r requirements.txt
RUN uv pip install --system -r requirements-dev.txt
RUN uv pip install --system -e .

# Run tests by default
CMD ["pytest", "-v", "--cov=pyterrier_generative", "--cov-report=term-missing"]
