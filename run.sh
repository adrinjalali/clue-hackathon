#!/bin/bash
echo "starting"

python3 src/save_binary.py ./data

PYTHONPATH=. python3 src/pipeline.py
