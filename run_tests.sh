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

# Check for libomp on Linux (common dependency for AdaptiveCpp CPU backend)
if [[ "$OSTYPE" == "linux-gnu"* ]] && command -v ldconfig >/dev/null 2>&1; then
    if ! ldconfig -p 2>/dev/null | grep -q "libomp"; then
        echo "WARNING: libomp (OpenMP) library not found. AdaptiveCpp CPU backend may fail."
        
        # Try to install if root
        if [ "$(id -u)" -eq 0 ] && command -v apt-get >/dev/null 2>&1; then
            echo "Attempting to install libomp-dev..."
            export DEBIAN_FRONTEND=noninteractive
            apt-get update -qq && apt-get install -y -qq libomp-dev || echo "WARNING: Install failed."
            # Update linker cache
            ldconfig
        fi

        # If still missing (or install failed), warn user
        if ! ldconfig -p 2>/dev/null | grep -q "libomp"; then
             echo "  - On Debian/Ubuntu: apt-get install libomp-dev"
        fi
    fi
fi

# Find libomp.so to ensure it's visible to AdaptiveCpp
LIBOMP_PATH=$(find /usr/lib /usr/local/lib -name "libomp.so" -o -name "libomp.so.5" -print -quit 2>/dev/null)
if [ -n "$LIBOMP_PATH" ]; then
    echo "Found libomp at: $LIBOMP_PATH"
    LIBOMP_DIR=$(dirname "$LIBOMP_PATH")
    if [[ ":$LD_LIBRARY_PATH:" != *":$LIBOMP_DIR:"* ]]; then
        export LD_LIBRARY_PATH="$LIBOMP_DIR:$LD_LIBRARY_PATH"
        echo "Added $LIBOMP_DIR to LD_LIBRARY_PATH"
    fi
else
    echo "WARNING: libomp.so not found. AdaptiveCpp CPU backend may fail."
fi

# Debug: Check dependencies of the installed extension
H3_TURBO_LOC=$(python3 -c "import h3_turbo; print(h3_turbo.__file__)" 2>/dev/null || true)
if [ -n "$H3_TURBO_LOC" ]; then
    echo "Checking dependencies of: $H3_TURBO_LOC"
    ldd "$H3_TURBO_LOC" | grep -i "omp\|cuda\|sycl\|acpp" || true
fi

# Enable AdaptiveCpp runtime logging if DEBUG is set (or unconditionally for diagnosis)
export ACPP_DEBUG_LEVEL=1
export OMP_NUM_THREADS=1
echo "--- AdaptiveCpp Debug Logging Enabled (Level 1) ---"
echo "--- GPU Status ---"
nvidia-smi || echo "WARNING: nvidia-smi failed. GPU may not be accessible."
echo "------------------"

echo "Running tests with PYTHONPATH=$PYTHONPATH"
python3 -m pytest -s -v tests/
