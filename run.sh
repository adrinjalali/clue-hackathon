#!/bin/bash
echo "starting"

python3 src/save_binary.py ./data

python3 src/pipeline.py
