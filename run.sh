#!/bin/bash
echo "starting"

python3 src/logistics.py ./data

python3 src/pipeline.py ./data
