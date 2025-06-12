#!/bin/bash

# RunPod Hugging Face Server Setup Script
# This script sets up the environment for the HF model server

set -e  # Exit on any error

echo "🚀 Setting up Hugging Face Model Server on RunPod"
echo "=================================================="

# Navigate to the project directory
cd /root/server/hf_container

# Check if Python is available
echo "🐍 Checking Python installation..."
python3 --version
which python3

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (compatible with your CUDA 12.8.1)
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install Transformers and related HuggingFace libraries
echo "🤗 Installing Transformers and HuggingFace libraries..."
pip install transformers
pip install accelerate
pip install datasets
pip install tokenizers

# Install FastAPI and related web server dependencies
echo "🌐 Installing FastAPI and web server dependencies..."
pip install fastapi
pip install uvicorn[standard]
pip install python-multipart
pip install pydantic

# Install additional utilities
echo "🛠️  Installing additional utilities..."
pip install psutil
pip install numpy
pip install scipy

# Install optional but useful packages
echo "📊 Installing optional packages..."
pip install bitsandbytes  # For quantization
pip install sentencepiece  # For some tokenizers
pip install protobuf

# Verify installations
echo "🔍 Verifying installations..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')" 
python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python3 -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"

# Check GPU
echo "🔥 Checking GPU availability..."
nvidia-smi

# Create a requirements.txt file for future reference
echo "📝 Creating requirements.txt..."
pip freeze > requirements.txt

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "source /root/server/hf_container/venv/bin/activate"
echo ""
echo "To start the server, run:"
echo "python3 hf_server.py --model <model_name>"
echo ""
echo "Example:"
echo "python3 hf_server.py --model microsoft/DialoGPT-medium"