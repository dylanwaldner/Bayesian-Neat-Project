#!/bin/bash
#This is a dummy app.sh file with the OpenAI key redacted. Generate your own API key, add to string, and rename sh file to app.sh to run project
source ~/.bashrc
source /lusr/opt/miniconda/bin/activate pytorch-cuda
pip install openai pyro-ppl neat

# Check if app.py exists
if [ -f "app.py" ]; then
    # Run the app.py file
    echo "Running app.py..."
    export CUDA_LAUNCH_BLOCKING=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OPENAI_API_KEY="<REDACTED>"
    python3 app.py
else
    echo "Error: app.py not found in the current directory."
fi

