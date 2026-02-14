#!/bin/bash
set -e

# Ensure PYTHONPATH includes the build directory where h3_turbo.so resides
export PYTHONPATH="$(pwd)/build:$PYTHONPATH"

# Check for h3 module and install requirements if missing (e.g. inside a new venv)
if ! python3 -c "import h3" &> /dev/null; then
    echo "Module 'h3' not found. Installing dependencies from requirements.txt..."
    python3 -m pip install -r requirements.txt
fi

# Enable AdaptiveCpp runtime logging if DEBUG is set
if [ -n "$DEBUG" ]; then
    export ACPP_DEBUG_LEVEL=3
    echo "--- AdaptiveCpp Debug Logging Enabled (Level 3) ---"
fi

echo "Running tests with PYTHONPATH=$PYTHONPATH"
python3 -m pytest -s -v tests/
