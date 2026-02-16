import pytest
import os
import sys
import datetime

# Add project root and build directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
build_dir = os.path.join(project_root, "build")

if project_root not in sys.path:
    sys.path.append(project_root)
if build_dir not in sys.path:
    sys.path.append(build_dir)

import license_generator

try:
    import h3_turbo
except ImportError:
    h3_turbo = None

@pytest.mark.skipif(h3_turbo is None, reason="h3_turbo module not found")
def test_license_valid(capfd):
    """Tests that a valid (future) license key is accepted."""
    user_id = "TEST_USER"
    future_date = (datetime.date.today() + datetime.timedelta(days=30)).strftime("%Y%m%d")
    valid_key = license_generator.generate_license(user_id, future_date)
    
    print(f"Testing valid key: {valid_key}")
    h3_turbo.set_license_key(valid_key)
    
    out, err = capfd.readouterr()

    with capfd.disabled():
        sys.stdout.write(out)
        sys.stderr.write(err)
    
    assert "License verified for: TEST_USER" in out
    assert f"Expires: {future_date}" in out

@pytest.mark.skipif(h3_turbo is None, reason="h3_turbo module not found")
def test_license_expired(capfd):
    """Tests that an expired (past) license key is rejected."""
    user_id = "TEST_USER"
    past_date = (datetime.date.today() - datetime.timedelta(days=30)).strftime("%Y%m%d")
    expired_key = license_generator.generate_license(user_id, past_date)
    
    print(f"Testing expired key: {expired_key}")
    h3_turbo.set_license_key(expired_key)
    
    out, err = capfd.readouterr()

    with capfd.disabled():
        sys.stdout.write(out)
        sys.stderr.write(err)
    
    assert "License expired for: TEST_USER" in err
    assert f"Expired: {past_date}" in err