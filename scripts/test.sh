#!/bin/bash

echo "🧪 Running tests for Databricks AWS Migration Tool..."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Run different types of tests
echo "Running unit tests..."
pytest tests/unit/ -v --cov=src --cov-report=html

echo "Running integration tests..."
pytest tests/integration/ -v

echo "Running end-to-end tests..."
pytest tests/e2e/ -v

echo "Generating coverage report..."
coverage html

echo "Tests completed! Check htmlcov/index.html for coverage report."
