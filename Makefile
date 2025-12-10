# ============================================
# Credit Risk API - Makefile
# Common commands for development and deployment
# ============================================

.PHONY: help install install-dev run docker-build docker-run deploy clean

# Default target
help:
	@echo "Credit Risk Prediction API - Available Commands"
	@echo "================================================"
	@echo ""
	@echo "Development:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make run          - Run API locally"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container locally"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy       - Deploy to Google Cloud Run"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean        - Remove cache and build files"
	@echo "  make test         - Run tests"

# Install production dependencies
install:
	pip install -r requirements-prod.txt

# Install development dependencies
install-dev:
	pip install -r requirements-dev.txt

# Run locally
run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Build Docker image
docker-build:
	docker build -t credit-risk-api .

# Run Docker container
docker-run:
	docker run -p 8080:8080 credit-risk-api

# Deploy to GCP
deploy:
	chmod +x deploy.sh && ./deploy.sh

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.log" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/ 2>/dev/null || true

# Run tests
test:
	pytest tests/ -v
