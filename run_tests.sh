#!/bin/bash
set -e

# Default to using installed package
USE_LOCAL_BUILD=0

# Parse args
for arg in "$@"; do
    if [[ "$arg" == "--local" ]]; then
        USE_LOCAL_BUILD=1
    fi
done

if [ "$USE_LOCAL_BUILD" -eq 1 ]; then
    if [ -d "build" ]; then
        echo "Forcing use of local build directory..."
        export PYTHONPATH="$(pwd)/build:$PYTHONPATH"
    else
        echo "Error: --local specified but 'build' directory not found."
        exit 1
    fi
else
    # Check if h3_turbo is installed
    if python3 -c "import h3_turbo" &> /dev/null; then
        echo "Testing installed 'h3_turbo' package."
    elif [ -f "build/h3_turbo.so" ] || [ -f "build/h3_turbo.dylib" ]; then
        echo "Package not installed. Falling back to local build in 'build/'..."
        export PYTHONPATH="$(pwd)/build:$PYTHONPATH"
    else
        echo "Error: h3_turbo not found in site-packages or build/ directory."
        echo "  - To build locally: ./build_app.sh"
        echo "  - To install wheel: pip install dist/*.whl"
        exit 1
    fi
fi

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

# Check RAM
TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
# 20GB = 20 * 1024 * 1024 = 20971520 KB
THRESHOLD_KB=20971520

SKIP_ARGS=""
if [ "$TOTAL_MEM_KB" -lt "$THRESHOLD_KB" ]; then
    echo "Total RAM (${TOTAL_MEM_KB} kB) is less than 20GB. Skipping tests/test_spatial_bench_q11_1B.py"
    SKIP_ARGS="--ignore=tests/test_spatial_bench_q11_1B.py"
else
    echo "Total RAM (${TOTAL_MEM_KB} kB) >= 20GB. Running all tests."
fi

echo "Running tests with PYTHONPATH=$PYTHONPATH"
python3 -m pytest -s -v tests/ $SKIP_ARGS
