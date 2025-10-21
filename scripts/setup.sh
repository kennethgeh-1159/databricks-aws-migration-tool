#!/bin/bash

echo "🚀 Setting up Databricks AWS Migration Tool development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    print_error "Python $python_version is installed, but Python $required_version or higher is required."
    exit 1
fi

print_status "Python $python_version detected ✓"

# Create virtual environment
print_status "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_status "Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies
print_status "Installing development dependencies..."
pip install -e ".[dev]"

# Create necessary directories
print_status "Creating project directories..."
mkdir -p logs data/results data/sample

# Copy environment template
if [ ! -f .env ]; then
    print_status "Creating .env file from template..."
    cp .env.example .env
    print_warning "Please edit .env file with your configuration"
fi

# Install pre-commit hooks
print_status "Installing pre-commit hooks..."
pre-commit install

# Run initial tests
print_status "Running initial tests..."
pytest tests/ -v

print_status "Setup completed successfully! 🎉"
print_status ""
print_status "Next steps:"
print_status "1. Edit .env file with your Databricks and AWS credentials"
print_status "2. Run 'source venv/bin/activate' to activate the virtual environment"
print_status "3. Run 'streamlit run app.py' to start the application"
print_status "4. Open http://localhost:8501 in your browser"
