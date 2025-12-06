#!/bin/bash

# Start the server in the background
if lsof -ti :8000 > /dev/null; then
    echo "Killing existing server on port 8000"
    kill -9 $(lsof -ti :8000)
fi
uvicorn server:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# Function to clean up background processes
cleanup() {
    echo "Stopping server..."
    kill $SERVER_PID
    exit
}

# Trap script exit signals
trap cleanup SIGINT SIGTERM

# Wait for the server to be ready
echo "Waiting for server to start..."
while ! curl -s "http://localhost:8000/" > /dev/null; do
    sleep 1
done

echo "Server started. Starting client..."
# Start the client
python client.py

#cleanup on exit 
cleanup
        print(f" Error connecting to the server at {SERVER_URL}.") 