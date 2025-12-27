.PHONY: help install install-dev format lint check test clean run build

# Default target
help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies (without dev)"
	@echo "  make install-dev   - Install all dependencies including dev"
	@echo "  make format        - Format code with black and isort"
	@echo "  make lint          - Lint code with flake8"
	@echo "  make check         - Format and lint code"
	@echo "  make test          - Run tests with coverage (requires 60% minimum)"
	@echo "  make clean         - Remove build artifacts and cache"
	@echo "  make build         - Build the project (flake8 check)"
	@echo "  make run           - Run example: TCPD APPLE Window L1 LSTM"

# Install dependencies (production only)
install:
	poetry install

# Install all dependencies including dev
install-dev:
	poetry install --with dev

# Format code with black and isort
format:
	@echo "🎨 Formatting code with black..."
	poetry run black src/
	@echo "📦 Sorting imports with isort..."
	poetry run isort src/
	@echo "✅ Code formatted!"

# Lint code with flake8
lint:
	@echo "🔍 Linting code with flake8..."
	poetry run flake8 src/
	@echo "✅ Code linted!"

# Run tests with pytest and coverage (fails if coverage < 60%)
test:
	@echo "🧪 Running tests with coverage..."
	poetry run pytest --cov=src --cov=config --cov-report=term-missing --cov-report=html --cov-fail-under=60
	@echo "✅ Tests completed! Coverage report available in htmlcov/index.html"

# Format, lint and test
build: format lint test
	@echo "✅ Code formatted and linted successfully!"

# Clean build artifacts and cache
clean:
	@echo "🧹 Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Run example command
run:
	@echo "🚀 Running example: TCPD APPLE Window L1 LSTM"
	poetry run python main.py TCPD APPLE Window L1 LSTM

# Run with custom arguments (use: make run-custom ARGS="TCPD APPLE Window L1 transformer")
run-custom:
	@echo "🚀 Running: $(ARGS)"
	poetry run python main.py $(ARGS)

# Show project info
info:
	@echo "📊 Project Information:"
	@echo "  Python version: $(shell poetry run python --version)"
	@echo "  Poetry version: $(shell poetry --version)"
	@echo "  Virtual environment: $(shell poetry env info --path)"
	@poetry show --tree | head -20
