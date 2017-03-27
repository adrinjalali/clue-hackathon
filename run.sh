#!/bin/bash
echo "starting"

python3 src/save_binary.py ./data

JOBLIB_TEMP_FOLDER=/tmp PYTHONPATH=. python3 src/pipeline.py
