#!/usr/bin/env bash

# Setup script for ALS Recommendation System
# This script sets up the development environment

set -e

echo "Setting up ALS Recommendation System..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.10+ is required. Found: $python_version"
    exit 1
fi

echo "Python version: $python_version ✓"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "Installing development dependencies..."
pip install pytest black ruff mypy

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw data/processed models assets/plots

# Run tests
echo "Running tests..."
python -m pytest tests/ -v

# Run linting
echo "Running linting..."
ruff check src/ scripts/ tests/
black --check src/ scripts/ tests/

echo "Setup complete! 🎉"
echo ""
echo "To activate the environment: source venv/bin/activate"
echo "To run the demo: streamlit run scripts/demo.py"
echo "To train models: python scripts/train.py"
