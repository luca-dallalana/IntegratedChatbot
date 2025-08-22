#!/bin/bash

echo "Testing Phase 2: API + Streamlit"
echo "================================"

# Test API health
echo "1. Testing API health..."
api_response=$(curl -s http://localhost:8501/ || echo "FAILED")
if [[ $api_response == *"Investment Chatbot"* ]]; then
    echo "✅ API responding"
else
    echo "❌ API not responding"
fi

# Test API docs
echo "2. Testing API documentation..."
docs_status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/docs)
if [[ $docs_status == "200" ]]; then
    echo "✅ API docs accessible"
else
    echo "❌ API docs not accessible"
fi

# Test test-cases endpoint
echo "3. Testing test-cases endpoint..."
cases_response=$(curl -s http://localhost:8000/test-cases || echo "FAILED")
if [[ $cases_response == *"test_cases"* ]]; then
    echo "✅ Test cases endpoint working"
else
    echo "❌ Test cases endpoint failed"
fi

# Test Streamlit
echo "4. Testing Streamlit..."
streamlit_status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501)
if [[ $streamlit_status == "200" ]]; then
    echo "✅ Streamlit accessible"
else
    echo "❌ Streamlit not accessible"
fi

echo ""
echo "Phase 2 verification complete!"
echo "API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "Streamlit: http://localhost:8501"