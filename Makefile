# Makefile for pyterrier-generative
# Mirrors GitHub Actions workflows for local development

.PHONY: all help install install-dev clean lint test test-unit test-integration test-coverage build docker-test docker-build docker-clean

# Default target
all: lint test-unit

# Help target
help:
	@echo "PyTerrier Generative - Makefile Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo ""
	@echo "Quality Checks (runs in CI):"
	@echo "  make lint             Run ruff linter (style.yml workflow)"
	@echo "  make test             Run all tests with coverage (test.yml workflow)"
	@echo "  make test-unit        Run unit tests only (exclude integration)"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-coverage    Run tests with HTML coverage report"
	@echo ""
	@echo "CI Simulation:"
	@echo "  make all              Run lint + test-unit (quick CI check)"
	@echo "  make ci               Run complete CI suite (lint + test)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-test      Run tests in Docker (mirrors CI environment)"
	@echo "  make docker-build     Build Docker image"
	@echo "  make docker-clean     Remove Docker containers and images"
	@echo ""
	@echo "Build & Deploy:"
	@echo "  make build            Build distribution packages (deploy.yml workflow)"
	@echo "  make clean            Clean build artifacts"
	@echo ""

# Installation targets
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

install-dev: install
	pip install -r requirements-dev.txt

# Linting (mirrors .github/workflows/style.yml)
lint:
	@echo "==> Running Ruff linter..."
	ruff check --output-format=github pyterrier_generative

lint-fix:
	@echo "==> Running Ruff with auto-fix..."
	ruff check --fix pyterrier_generative

# Testing (mirrors .github/workflows/test.yml)
test:
	@echo "==> Running tests with coverage..."
	pytest --durations=20 \
		-p no:faulthandler \
		--json-report \
		--json-report-file test.results.json \
		--cov pyterrier_generative \
		--cov-report json:test.coverage.json \
		--cov-report term-missing \
		tests/

test-unit:
	@echo "==> Running unit tests (excluding integration)..."
	pytest -v \
		--cov=pyterrier_generative \
		--cov-report=term-missing \
		-m "not integration" \
		tests/

test-integration:
	@echo "==> Running integration tests..."
	pytest -v \
		--cov=pyterrier_generative \
		--cov-report=term-missing \
		-m "integration" \
		tests/

test-coverage:
	@echo "==> Running tests with HTML coverage report..."
	pytest -v \
		--cov=pyterrier_generative \
		--cov-report=html \
		--cov-report=term-missing \
		tests/
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

test-fast:
	@echo "==> Running fast tests (no coverage)..."
	pytest -v -m "not integration" tests/

# Docker targets (uses docker-compose.yml)
docker-build:
	@echo "==> Building Docker image..."
	docker-compose build test

docker-test: docker-build
	@echo "==> Running tests in Docker (mirrors CI environment)..."
	docker-compose run --rm test

docker-test-all: docker-build
	@echo "==> Running all tests in Docker (including integration)..."
	docker-compose run --rm test-all

docker-shell: docker-build
	@echo "==> Starting interactive Docker shell..."
	docker-compose run --rm shell

docker-clean:
	@echo "==> Cleaning Docker containers and images..."
	docker-compose down --rmi local --volumes --remove-orphans

# Build targets (mirrors .github/workflows/deploy.yml)
build:
	@echo "==> Building distribution packages..."
	python -m pip install --upgrade pip
	pip install setuptools wheel twine build
	python -m build
	@echo ""
	@echo "Distribution packages built in dist/"

# CI simulation
ci: lint test
	@echo ""
	@echo "==> CI checks completed successfully!"

# Clean targets
clean:
	@echo "==> Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .ruff_cache/
	rm -rf *.results.json
	rm -rf *.coverage.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete"

clean-all: clean docker-clean
	@echo "==> Deep clean complete (including Docker)"

# Development helpers
watch-test:
	@echo "==> Watching for changes and running tests..."
	@echo "Requires: pip install pytest-watch"
	ptw -- -v -m "not integration" tests/

format:
	@echo "==> Auto-formatting code with ruff..."
	ruff format pyterrier_generative tests/

check: lint test-unit
	@echo ""
	@echo "==> Pre-commit checks passed!"
