import pytest
import os
import sys

# Add project root to path to allow importing license_generator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Add build directory to path to allow importing h3_turbo if not installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))

# Try to import h3_turbo. If it fails, we can't set the license anyway.
try:
    import h3_turbo
except ImportError:
    h3_turbo = None

# Suppress AdaptiveCpp warnings (like missing OpenCL) unless they are errors
if "ACPP_LOG_LEVEL" not in os.environ:
    os.environ["ACPP_LOG_LEVEL"] = "error"

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