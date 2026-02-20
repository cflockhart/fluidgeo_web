#!/bin/bash
set -e

echo "--- Configuring environment for h3_turbo Jupyter Lab ---"

# 1. Check for h3 module and install requirements if missing (e.g. inside a new venv)
if ! python3 -c "import h3" &> /dev/null; then
    if [ -f "requirements.txt" ]; then
        echo "Module 'h3' not found. Installing dependencies from requirements.txt..."
        python3 -m pip install -r requirements.txt
    fi
fi

# 2. Handle libomp (OpenMP) for AdaptiveCpp CPU backend
# AdaptiveCpp often relies on LLVM's OpenMP runtime.
if [[ "$OSTYPE" == "linux-gnu"* ]] && command -v ldconfig >/dev/null 2>&1; then
    if ! ldconfig -p 2>/dev/null | grep -q "libomp"; then
        echo "WARNING: libomp (OpenMP) library not found."
        
        # Try to install if root (common in RunPod/Docker)
        if [ "$(id -u)" -eq 0 ] && command -v apt-get >/dev/null 2>&1; then
            echo "Attempting to install libomp-dev..."
            export DEBIAN_FRONTEND=noninteractive
            apt-get update -qq && apt-get install -y -qq libomp-dev
            ldconfig
        else
            echo "  Please install libomp-dev manually."
        fi
    fi
fi

# Find libomp.so and add to LD_LIBRARY_PATH
LIBOMP_PATH=$(find /usr/lib /usr/local/lib -name "libomp.so" -o -name "libomp.so.5" -print -quit 2>/dev/null)
if [ -n "$LIBOMP_PATH" ]; then
    echo "Found libomp at: $LIBOMP_PATH"
    LIBOMP_DIR=$(dirname "$LIBOMP_PATH")
    if [[ ":$LD_LIBRARY_PATH:" != *":$LIBOMP_DIR:"* ]]; then
        export LD_LIBRARY_PATH="$LIBOMP_DIR:$LD_LIBRARY_PATH"
        echo "Updated LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
    fi
else
    echo "WARNING: libomp.so not found. AdaptiveCpp CPU backend may fail."
fi

# 3. Set AdaptiveCpp Environment Variables
export ACPP_DEBUG_LEVEL=1
export OMP_NUM_THREADS=1

echo "Environment variables set:"
echo "  ACPP_DEBUG_LEVEL=$ACPP_DEBUG_LEVEL"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"

echo "Starting Jupyter Lab..."
exec jupyter lab --allow-root --ip=0.0.0.0 --no-browser --port=8888 "$@"