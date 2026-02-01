#!/bin/bash
# Startup script for Survey Bot Web UI with conda tf environment

# Activate conda
source "D:\Study\anaconda3\etc\profile.d\conda.sh"

# Activate tf environment
conda activate tf

# Navigate to project directory
cd "J:\Project Files\MyPython\autoReceipt"

# Start the web server
python run_web.py
