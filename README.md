## Benchmark files and example usage for h3_turbo

* > 1000x speedup on an RTX PRO 6000!*
## Installation instructions
Only tested on Ubuntu so far. Please send us the error if installation or tests fail on other distros.

- Clone the repo
- Requires CUDA toolkit to be installed: `sudo apt update && sudo apt install nvidia-cuda-toolkit`

- Pick the "manylinux" wheel file from the `dist` directory (`dbr` files are specifically for Databricks runtime configurations), according to your Python version:
 - Ubuntu 24.04: h3_turbo-0.0.1-cp312-cp312-manylinux_2_39_x86_64.whl
 - Ubuntu 22.04: h3_turbo-0.0.1-cp311-cp311-manylinux_2_39_x86_64.whl

Run `pip install >>wheel file<<`
Only tested on Ubuntu 24.04 and 22.04 with NVIDIA sm_89 and up architecture (4000 series, 5000 series, L4, RTX 6000 etc). You can find the compute_cap with `nvidia-smi --query-gpu=compute_cap`. If it's 8.9 or higher, it should work.

## RTX PRO 6000 Blackwell server edition (AWS G7e Instances)
``` 
================================================== test session starts ==================================================
platform linux -- Python 3.12.3, pytest-9.0.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /workspace/h3-sycl-bridge
configfile: pyproject.toml
plugins: anyio-4.11.0
collected 5 items

tests/test_benchmark.py::test_heatmap_performance
[conftest] Auto-generating license key for test session...
License verified for: pytest_auto (Expires: 20991231)

warming up the JIT...
DEBUG: h3_turbo loaded from /workspace/h3-sycl-bridge/build/h3_turbo.so
[AdaptiveCpp Warning] from /workspace/h3-sycl-bridge/AdaptiveCpp/src/runtime/ocl/ocl_hardware_manager.cpp:626 @ ocl_hardware_manager(): ocl_hardware_manager: Could not obtain platform list (error code = CL:-1001)
SYCL Device: NVIDIA RTX PRO 6000 Blackwell Server Edition
IT Ready!

Generating 10000000 random H3 indexes...
Performing heatmap generation on GPU...
GPU heatmap took: 0.0051 seconds.
GPU Throughput: 1,943,967,371 points/sec

Performing heatmap generation on CPU...
CPU heatmap took: 7.3307 seconds.

# ============================================================
DOORDASH WORKLOAD: (10,000,000 Pings)
CPU Total Pipeline: 7.3307 seconds
GPU Total Pipeline: 0.0051 seconds
OVERALL SPEEDUP: 1425.07x

Verifying results...
✅ GPU and CPU results match.

---

## Benchmark complete. To run the tests, use the command: pytest

# PASSED
tests/test_raw_benchmark.py::test_raw_compute_benchmark

## RAW COMPUTE BENCHMARK: TRANSFORMING 50,000,000 H3 INDEXES
This test measures the ideal GPU workload (pure arithmetic) without overhead.

Generating data...
Generated 30301 valid unique seeds from k-ring.
Expanded to 50,000,000 items.
Running GPU benchmark...
GPU transformation complete in 0.0237 s
GPU Throughput: 2,106,060,636 points/sec

Running CPU benchmark (Vectorized NumPy)...
Step 1: Cell to Parent (using h3-py)...
Step 2: Scramble (using NumPy)...
CPU transformation complete in 36.5297 s

# ================================================================================
FINAL RESULTS
CPU Time: 36.5297 s
GPU Time: 0.0237 s
RAW COMPUTE SPEEDUP: 1538.67x

# Verification successful.
PASSED
tests/test_sf1000.py::test_q11_spatial_join

# SPATIALBENCH QUERY 11: SPATIAL JOIN
Joining 50,000,000 pings against 1,000,000 zones

Running GPU Spatial Join...
GPU Time: 0.0492 s
GPU Throughput: 1,016,239,267 points/sec
Running CPU Spatial Join (Baseline)...
CPU Time: 39.1718 s

SPEEDUP: 796.16x
Verification Passed.
PASSED
tests/test_spatial_benchmark.py::test_spatial_inclusion_performance
Generating 10000000 pings and 100000 hot zones...

Performing spatial inclusion check on GPU...
GPU spatial join took: 0.0138 seconds.

Performing spatial inclusion check on CPU...
CPU spatial join took: 7.8626 seconds.

# ============================================================
SPATIAL JOIN: (10,000,000 Pings against 100,000 Zones)
CPU Time: 7.8626 s
GPU Time: 0.0138 s
SPEEDUP: 568.53x

# Found 9999328 matches.
PASSED
tests/test_spatialbench_q11.py::test_q11_spatial_join

# [AdaptiveCpp Warning] from /workspace/h3-sycl-bridge/AdaptiveCpp/src/runtime/ocl/ocl_hardware_manager.cpp:626 @ ocl_hardware_manager(): ocl_hardware_manager: Could not obtain platform list (error code = CL:-1001)

# ================================================================================
SPATIALBENCH QUERY 11: SPATIAL JOIN
Joining 1,000,000,000 pings against 1,000,000 zones

Generating pings...
Warming up JIT...
[AdaptiveCpp Warning] from /workspace/h3-sycl-bridge/AdaptiveCpp/src/runtime/ocl/ocl_hardware_manager.cpp:626 @ ocl_hardware_manager(): ocl_hardware_manager: Could not obtain platform list (error code = CL:-1001)
SYCL Device: NVIDIA RTX PRO 6000 Blackwell Server Edition
Running GPU Spatial Join...
GPU Time: 0.5772 s
Running CPU Spatial Join (Baseline)...
CPU Time: 684.7718 s

SPEEDUP: 1186.28x
Verification Passed.
PASSED

============================================= 1 passed in 692.67s (0:11:32) =============================================
```
### NVIDIA L4 (AWS G6 instances)
```
root@33a57dec78f5 /w/h3-sycl-bridge (main)# ./run_tests.sh
Running tests with PYTHONPATH=/workspace/h3-sycl-bridge/build:
================================================== test session starts ==================================================
platform linux -- Python 3.12.12, pytest-9.0.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /workspace/h3-sycl-bridge
configfile: pyproject.toml
collected 5 items

tests/test_benchmark.py::test_heatmap_performancewarming up the JIT...
DEBUG: h3_turbo loaded from /workspace/h3-sycl-bridge/build/h3_turbo.so
[AdaptiveCpp Warning] from /workspace/h3-sycl-bridge/AdaptiveCpp/src/runtime/ocl/ocl_hardware_manager.cpp:626 @ ocl_hardware_manager(): ocl_hardware_manager: Could not obtain platform list (error code = CL:-1001)
SYCL Device: NVIDIA L4
JIT Ready!

Generating 10000000 random H3 indexes...
Performing heatmap generation on GPU...
GPU heatmap took: 0.0147 seconds.
GPU Throughput: 682,011,740 points/sec

Performing heatmap generation on CPU...
CPU heatmap took: 6.4051 seconds.

# ============================================================
DOORDASH WORKLOAD: (10,000,000 Pings)
CPU Total Pipeline: 6.4051 seconds
GPU Total Pipeline: 0.0147 seconds
OVERALL SPEEDUP: 436.83x

Verifying results...
✅ GPU and CPU results match.

---

## Benchmark complete. To run the tests, use the command: pytest

# PASSED
tests/test_raw_benchmark.py::test_raw_compute_benchmark

## RAW COMPUTE BENCHMARK: TRANSFORMING 50,000,000 H3 INDEXES
This test measures the ideal GPU workload (pure arithmetic) without overhead.

Generating data...
Generated 30301 valid unique seeds from k-ring.
Expanded to 50,000,000 items.
Running GPU benchmark...
GPU transformation complete in 0.0705 s
GPU Throughput: 708,784,642 points/sec

Running CPU benchmark (Vectorized NumPy)...
Step 1: Cell to Parent (using h3-py)...
Step 2: Scramble (using NumPy)...
CPU transformation complete in 30.3379 s

# ================================================================================
FINAL RESULTS
CPU Time: 30.3379 s
GPU Time: 0.0705 s
RAW COMPUTE SPEEDUP: 430.06x

# Verification successful.
PASSED
tests/test_sf1000.py::test_q11_spatial_join

# SPATIALBENCH QUERY 11: SPATIAL JOIN
Joining 50,000,000 pings against 1,000,000 zones

Preparing index structures...
Running GPU Spatial Join...
GPU Time: 0.0965 s
GPU Throughput: 517,994,368 points/sec
Running CPU Spatial Join (Baseline)...
CPU Time: 38.1490 s

SPEEDUP: 395.22x
Verification Passed.
PASSED
tests/test_spatial_benchmark.py::test_spatial_inclusion_performance
Generating 10000000 pings and 100000 hot zones...
Preparing hash table...
Hash table built in 0.0675s

Performing spatial inclusion check on GPU...
GPU spatial join took: 0.0222 seconds.

Performing spatial inclusion check on CPU...
CPU spatial join took: 7.7431 seconds.

# ============================================================
SPATIAL JOIN: (10,000,000 Pings against 100,000 Zones)
CPU Time: 7.7431 s
GPU Time: 0.0222 s
SPEEDUP: 348.33x

# Found 9999770 matches.
PASSED
tests/test_spatialbench_q11.py::test_q11_spatial_join

# SPATIALBENCH QUERY 11: SPATIAL JOIN
Joining 1,000,000,000 pings against 1,000,000 zones

Preparing index structures...
Running GPU Spatial Join...
GPU Time: 1.9403 s
Running CPU Spatial Join (Baseline)...
CPU Time: 1689.7005 s

SPEEDUP: 870.84x
Verification Passed.
PASSED

============================================ 5 passed in 2159.52s (0:35:59) =============================================
```
## NVIDIA RTX 4090
```

================================================== test session starts ==================================================
platform linux -- Python 3.12.12, pytest-9.0.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /workspace/h3-sycl-bridge
configfile: pyproject.toml
collected 5 items

tests/test_benchmark.py::test_heatmap_performance
warming up the JIT...
DEBUG: h3_turbo loaded from /workspace/h3-sycl-bridge/build/h3_turbo.cpython-312-x86_64-linux-gnu.so
[AdaptiveCpp Warning] from /workspace/h3-sycl-bridge/AdaptiveCpp/src/runtime/ocl/ocl_hardware_manager.cpp:626 @ ocl_hardware_manager(): ocl_hardware_manager: Could not obtain platform list (error code = CL:-1001)
SYCL Device: NVIDIA GeForce RTX 4090
JIT Ready!

Generating 10000000 random H3 indexes...
Performing heatmap generation on GPU...
GPU heatmap took: 0.0141 seconds.
GPU Throughput: 711,248,580 points/sec

Performing heatmap generation on CPU...
CPU heatmap took: 6.5639 seconds.

# ============================================================
DOORDASH WORKLOAD: (10,000,000 Pings)
CPU Total Pipeline: 6.5639 seconds
GPU Total Pipeline: 0.0141 seconds
OVERALL SPEEDUP: 466.86x

Verifying results...
✅ GPU and CPU results match.

---

## Benchmark complete. To run the tests, use the command: pytest

# PASSED
tests/test_raw_benchmark.py::test_raw_compute_benchmark

## RAW COMPUTE BENCHMARK: TRANSFORMING 50,000,000 H3 INDEXES
This test measures the ideal GPU workload (pure arithmetic) without overhead.

Generating data...
Generated 30301 valid unique seeds from k-ring.
Expanded to 50,000,000 items.
Running GPU benchmark...
GPU transformation complete in 0.0673 s
GPU Throughput: 742,791,162 points/sec

Running CPU benchmark (Vectorized NumPy)...
Step 1: Cell to Parent (using h3-py)...
Step 2: Scramble (using NumPy)...
CPU transformation complete in 31.5578 s

# ================================================================================
FINAL RESULTS
CPU Time: 31.5578 s
GPU Time: 0.0673 s
RAW COMPUTE SPEEDUP: 468.82x

# Verification successful.
PASSED
tests/test_sf1000.py::test_q11_spatial_join

# SPATIALBENCH QUERY 11: SPATIAL JOIN
Joining 50,000,000 pings against 1,000,000 zones

Running GPU Spatial Join...
GPU Time: 0.1055 s
GPU Throughput: 473,713,195 points/sec
Running CPU Spatial Join (Baseline)...
CPU Time: 38.9207 s

SPEEDUP: 368.74x
Verification Passed.
PASSED
tests/test_spatial_benchmark.py::test_spatial_inclusion_performance
Generating 10000000 pings and 100000 hot zones...

Performing spatial inclusion check on GPU...
GPU spatial join took: 0.0425 seconds.

Performing spatial inclusion check on CPU...
CPU spatial join took: 7.9469 seconds.

# ============================================================
SPATIAL JOIN: (10,000,000 Pings against 100,000 Zones)
CPU Time: 7.9469 s
GPU Time: 0.0425 s
SPEEDUP: 187.16x

# Found 10000000 matches.
PASSED
tests/test_spatialbench_q11.py::test_q11_spatial_join

# SPATIALBENCH QUERY 11: SPATIAL JOIN
Joining 110,000,000 pings against 100,000 zones

Generating pings...
Warming up JIT...
Running GPU Spatial Join...
GPU Time: 0.1292 s
Running CPU Spatial Join (Baseline)...
CPU Time: 62.1072 s

SPEEDUP: 480.78x
Verification Passed.
PASSED

============================================= 5 passed in 156.98s (0:02:36) =============================================
```
