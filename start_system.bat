@echo off
ECHO Starting Autonomous Traffic Manager System

ECHO Starting AI Backend...
START /B python ai_model/ai_api.py

ECHO Waiting for backend to initialize...
TIMEOUT /T 2 > NUL

ECHO Starting Frontend...
START /B npm start

ECHO Autonomous Traffic Manager system is running!
ECHO - Backend: http://localhost:5000
ECHO - Frontend: http://localhost:3000
ECHO.
ECHO Press Ctrl+C in each console window to shutdown components

PAUSE 