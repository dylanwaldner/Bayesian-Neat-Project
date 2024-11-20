#!/bin/bash
source /lusr/opt/miniconda/bin/activate pytorch-cuda
# Install pytest
pip install pytest

# Run the Python test
python3 test_bnn_neat_full.py

