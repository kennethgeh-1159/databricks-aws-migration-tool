#!/bin/bash

echo "🔨 Building Databricks AWS Migration Tool..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    print_status "Virtual environment activated"
else
    print_error "Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Format code
print_status "Formatting code with Black..."
black src/ tests/ app.py --line-length=88

# Sort imports
print_status "Sorting imports with isort..."
isort src/ tests/ app.py --profile black

# Lint code
print_status "Linting code with flake8..."
flake8 src/ tests/ app.py --max-line-length=88 --ignore=E203,W503

# Type checking
print_status "Type checking with mypy..."
mypy src/ --ignore-missing-imports

# Run tests
print_status "Running tests..."
pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Build package
print_status "Building package..."
python -m build

# Check package
print_status "Checking package..."
twine check dist/*

print_status "Build completed successfully! ✅"
