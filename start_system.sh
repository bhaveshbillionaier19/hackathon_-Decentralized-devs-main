#!/bin/bash

# Start the AI backend
echo "Starting AI Backend..."
cd ai_model
python ai_api.py &
BACKEND_PID=$!
cd ..

# Wait for backend to initialize
sleep 2
echo "Backend started with PID: $BACKEND_PID"

# Start the React frontend
echo "Starting Frontend..."
npm start &
FRONTEND_PID=$!

echo "Frontend started with PID: $FRONTEND_PID"
echo "Autonomous Traffic Manager system is running!"
echo "- Backend: http://localhost:5000"
echo "- Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to shutdown the system"

# Wait for user to stop the system
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait 