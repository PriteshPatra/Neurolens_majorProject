#!/bin/bash
# Neurolens Native Backend Launcher (Linux / Mac)
# Use this to run the FastAPI backend natively on your host OS.
# This gives the application direct access to DBus and Display controllers,
# allowing the automatic OS brightness and Native OS toast notifications to work properly.

echo "==============================================="
echo "   Neurolens Native OS Hardware Controller     "
echo "==============================================="

# Navigate to backend directory
cd "$(dirname "$0")/backend" || exit

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Verifying and installing dependencies..."
pip install -r requirements.txt

# Start the server
echo ""
echo "🚀 Starting Native FastAPI Backend on port 8001..."
echo "Your OS brightness and notifications are now directly connected!"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8001 --workers 6
