#!/bin/bash
echo "starting"
export JOBLIB_TEMP_FOLDER=/tmp

python3 src/save_binary.py ./data

PYTHONPATH=. python3 src/adrin-pipeline.py
