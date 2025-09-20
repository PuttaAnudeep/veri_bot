#!/bin/bash

# VeriBot Startup Script
# This script starts the VeriBot application with all necessary components

echo "🚀 Starting VeriBot - AI-Powered Data Visualization Platform"
echo "============================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 to continue."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 to continue."
    exit 1
fi

echo "✅ Python3 and pip3 are available"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📦 Installing Python dependencies..."
pip install --upgrade pip

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found, installing common packages..."
    pip install fastapi uvicorn pandas sqlalchemy psycopg2-binary python-dotenv plotly google-generativeai
fi

# Check if .env file exists
if [ ! -f "backend/.env" ]; then
    echo "⚠️ .env file not found in backend directory"
    echo "Please create backend/.env with the following variables:"
    echo "NEON_CONNECTION_STRING=your_postgresql_connection_string"
    echo "GEMINI_API_KEY=your_gemini_api_key"
    echo ""
    read -p "Do you want to continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Exiting. Please create the .env file and try again."
        exit 1
    fi
fi

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "❌ Backend directory not found!"
    exit 1
fi

# Change to backend directory
cd backend

echo "🔍 Checking backend files..."
required_files=("api_server.py" "advanced_query_processor.py" "chart_config.json")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Required file $file not found!"
        exit 1
    fi
done

echo "✅ All required backend files found"

# Check if frontend exists
if [ ! -d "../frontend" ]; then
    echo "⚠️ Frontend directory not found. API will still work but web interface won't be available."
else
    echo "✅ Frontend directory found"
fi

echo ""
echo "🎯 Starting VeriBot API Server..."
echo "📊 Features enabled:"
echo "   • Natural language to SQL conversion"
echo "   • AI-powered visualization suggestions"
echo "   • 10+ chart types available"
echo "   • Interactive data exploration"
echo ""
echo "🌐 The application will be available at:"
echo "   • Web Interface: http://localhost:8000"
echo "   • API Documentation: http://localhost:8000/docs"
echo "   • Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "============================================================"

# Start the API server
python api_server.py