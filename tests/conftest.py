import pytest
import os
import sys
import multiprocessing

# Add project root to path to allow importing license_generator
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Try to import h3_turbo first to prevent shadowing of installed packages
try:
    import h3_turbo
    import h3_turbo._h3_turbo
except ImportError:
    # Add build directory to path to allow importing h3_turbo if not installed
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
    # Add src directory to path to allow importing h3_turbo python wrapper
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Try to import h3_turbo. If it fails, we can't set the license anyway.
try:
    import h3_turbo
except ImportError:
    h3_turbo = None

# Suppress AdaptiveCpp warnings (like missing OpenCL) unless they are errors
if "ACPP_LOG_LEVEL" not in os.environ:
    os.environ["ACPP_LOG_LEVEL"] = "error"

@pytest.fixture(scope="session", autouse=True)
def set_multiprocessing_start_method():
    """
    Sets the multiprocessing start method to 'spawn' to avoid DeprecationWarning
    when the parent process (pytest) is multi-threaded.
    """
    # This must be called exactly once per process, before any processes are started.
    # 'spawn' is safer for multi-threaded parent processes.
    # Use a try-except block as it might already be set by other means or in non-main processes.
    try:
        # Only set if not already set to 'spawn'. Use force=True to override 'fork' if it's the default.
        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
            print("\n[conftest] Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"\n[conftest] Warning: Could not set multiprocessing start method: {e}")
    except Exception as e:
        print(f"\n[conftest] Unexpected error setting multiprocessing start method: {e}")

@pytest.fixture(scope="session", autouse=True)
def auto_configure_license():
    """
    Ensures a valid H3 Turbo license is set for the test session.
    If not present in environment, generates a temporary one.
    """
    if "H3_TURBO_LICENSE" not in os.environ:
        try:
            import license_generator
            print("\n[conftest] Auto-generating license key for test session...")
            # Generate a key valid for a long time
            key = license_generator.generate_license("pytest_auto", "20991231")
            os.environ["H3_TURBO_LICENSE"] = key
        except ImportError:
            print("\n[conftest] Warning: license_generator.py not found. Tests may fail if license is missing.")
    
    if "H3_TURBO_LICENSE" in os.environ and h3_turbo:
        key = os.environ["H3_TURBO_LICENSE"].strip()
        h3_turbo.set_license_key(key)