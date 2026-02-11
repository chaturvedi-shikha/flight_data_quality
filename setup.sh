#!/bin/bash

# Flight Data Quality Analysis - Quick Start Script

echo "========================================="
echo "Flight Data Quality Analysis - Setup"
echo "========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run tests (TDD):"
echo "   pytest -v"
echo "   pytest -v --cov=src --cov-report=term-missing"
echo ""
echo "3. Launch interactive dashboard:"
echo "   streamlit run ui/dashboard.py"
echo ""
echo "4. Generate command-line reports:"
echo "   python -c 'from src.data_quality import run_quality_check; run_quality_check(\"data/flight_data_2024_sample.csv\")'"
echo ""
echo "========================================="
