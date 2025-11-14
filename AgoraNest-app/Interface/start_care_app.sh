#!/bin/bash

# Comprehensive Care AI Web Application Startup Script
echo "🏠 Starting Comprehensive Care AI Web Application..."

# Check if we're in the right directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"

# Activate conda environment
echo "🔄 Activating conda environment 'care'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate care

# Check if model exists
MODEL_PATH="/hd1/Joydeep-multiagent/xagents/Fine-Models/Final-qwen3-elderly-care"
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found at $MODEL_PATH"
    echo "Please ensure the model is properly installed."
    exit 1
fi

# Create necessary directories
echo "📂 Creating necessary directories..."
mkdir -p ../memory
mkdir -p ../voice_data
mkdir -p web_data/voice_profiles
mkdir -p static/recordings
mkdir -p templates

# Set environment variables
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export FLASK_ENV=development

echo "🚀 Starting web application..."
echo "🌐 The application will be available at: http://localhost:5000"
echo "🔊 Make sure your microphone is connected and working"
echo "📱 The interface supports both voice and text input"
echo ""
echo "Press Ctrl+C to stop the application"
echo "================================================"

# Start the Flask application
python web_app.py
