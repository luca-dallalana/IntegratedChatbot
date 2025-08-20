#!/bin/bash

# Investment Chatbot System - Complete Setup and Run Script
# This script sets up and runs the entire investment chatbot comparison system

set -e  # Exit on any error

echo "🚀 Investment Chatbot System Setup"
echo "=================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8 or higher and try again."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📋 Installing Python dependencies..."
pip install streamlit fastapi uvicorn openai pandas plotly numpy requests pydantic python-multipart websockets asyncio statistics

echo "✅ All dependencies installed successfully"

# Check for required files
echo "📁 Checking required files..."

if [ ! -f "integrated_investment_system.py" ]; then
    echo "❌ Missing: integrated_investment_system.py"
    echo "Please save the main system file from the artifacts."
    exit 1
fi

if [ ! -f "investmentChatBot.html" ]; then
    echo "❌ Missing: investmentChatBot.html"
    echo "Please save your HTML chatbot file."
    exit 1
fi

if [ ! -f "metrics_dashboard.py" ]; then
    echo "❌ Missing: metrics_dashboard.py"
    echo "Please save the metrics dashboard file from the artifacts."
    exit 1
fi

echo "✅ All required files found"

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️ OpenAI API key not found in environment variables"
    echo "You'll need to enter it in the web interface"
else
    echo "✅ OpenAI API key found in environment"
fi

echo ""
echo "🎯 Setup complete! Choose how to run the system:"
echo "================================================"
echo ""
echo "1. Streamlit Dashboard (Recommended for testing)"
echo "   - Complete UI with advanced metrics"
echo "   - Model comparison and analysis"
echo "   - Run: streamlit run integrated_investment_system.py streamlit"
echo ""
echo "2. API + HTML Chatbot (Production setup)"
echo "   - Your original chatbot interface"
echo "   - API backend for comparisons"
echo "   - Run API: python integrated_investment_system.py api"
echo "   - Run HTML: python -m http.server 3000"
echo ""
echo "3. Full Integration (All services)"
echo "   - API + Streamlit + HTML all running"
echo "   - Use this script with 'full' parameter"
echo ""

# Parse command line argument
if [ "$1" = "streamlit" ]; then
    echo "🌟 Starting Streamlit Dashboard..."
    echo "Navigate to: http://localhost:8501"
    echo "Press Ctrl+C to stop"
    echo ""
    streamlit run integrated_investment_system.py streamlit

elif [ "$1" = "api" ]; then
    echo "🔌 Starting API Backend..."
    echo "API available at: http://localhost:8000"
    echo "API docs at: http://localhost:8000/docs"
    echo "Press Ctrl+C to stop"
    echo ""
    python integrated_investment_system.py api

elif [ "$1" = "html" ]; then
    echo "🌐 Starting HTML Server..."
    echo "Chatbot available at: http://localhost:3000"
    echo "Press Ctrl+C to stop"
    echo ""
    python -m http.server 3000

elif [ "$1" = "full" ]; then
    echo "🎛️ Starting Full Integration..."
    echo ""
    echo "Services starting:"
    echo "- API Backend: http://localhost:8000"
    echo "- Streamlit Dashboard: http://localhost:8501"
    echo "- HTML Chatbot: http://localhost:3000"
    echo ""
    echo "Press Ctrl+C to stop all services"
    echo ""
    
    # Start API in background
    python integrated_investment_system.py api &
    API_PID=$!
    
    # Start HTML server in background
    python -m http.server 3000 &
    HTML_PID=$!
    
    # Wait a moment for services to start
    sleep 3
    
    # Start Streamlit (foreground)
    streamlit run integrated_investment_system.py streamlit &
    STREAMLIT_PID=$!
    
    # Wait for user interrupt
    trap 'echo ""; echo "🛑 Stopping all services..."; kill $API_PID $HTML_PID $STREAMLIT_PID 2>/dev/null; exit 0' INT
    
    wait

elif [ "$1" = "test" ]; then
    echo "🧪 Running System Tests..."
    echo ""
    
    # Test API startup
    echo "Testing API startup..."
    timeout 10s python integrated_investment_system.py api &
    API_TEST_PID=$!
    sleep 5
    
    if kill -0 $API_TEST_PID 2>/dev/null; then
        echo "✅ API starts successfully"
        kill $API_TEST_PID
    else
        echo "❌ API failed to start"
    fi
    
    # Test Streamlit
    echo "Testing Streamlit startup..."
    timeout 10s streamlit run integrated_investment_system.py streamlit --server.headless true &
    STREAMLIT_TEST_PID=$!
    sleep 5
    
    if kill -0 $STREAMLIT_TEST_PID 2>/dev/null; then
        echo "✅ Streamlit starts successfully"
        kill $STREAMLIT_TEST_PID
    else
        echo "❌ Streamlit failed to start"
    fi
    
    echo "✅ System tests completed"

else
    echo "Usage: $0 [streamlit|api|html|full|test]"
    echo ""
    echo "Quick start examples:"
    echo "  $0 streamlit    # Start dashboard for testing"
    echo "  $0 api          # Start API backend only"
    echo "  $0 html         # Start HTML chatbot only"
    echo "  $0 full         # Start all services"
    echo "  $0 test         # Run system tests"
    echo ""
    echo "Or run services manually:"
    echo "  streamlit run integrated_investment_system.py streamlit"
    echo "  python integrated_investment_system.py api"
    echo "  python -m http.server 3000"
fi