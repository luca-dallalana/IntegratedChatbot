#!/bin/bash

# Function to kill all background processes
cleanup() {
    echo "Cleaning up background processes..."
    pkill -P $$
    exit
}

# Trap exit signals to run cleanup function
trap cleanup EXIT

# Start the API server in the background
echo "Starting FastAPI server..."
python3 src/main.py api &
API_PID=$!

# Wait a moment for the server to start
sleep 3

# Start the Streamlit app in the foreground
echo "Starting Streamlit app..."
streamlit run src/main.py streamlit

# Wait for all background jobs to finish (they won't, but it's good practice)
wait
