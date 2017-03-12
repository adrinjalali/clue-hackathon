#!/usr/bin/env bash
# Checks the necessary files are inside the container.
set -e

if ! [ -e "/run.sh" ]; then
    echo "/run.sh not found inside container." 1>&2
    echo "Is it correctly added to the container?" 1>&2
    exit 1
fi

