# FluidGeo H3-Turbo

[![License](https://img.shields.io/badge/license-Dual--License-blue.svg)](#licensing)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/h3-turbo/)
[![Platform](https://img.shields.io/badge/platform-linux--64-lightgrey)](https://pypi.org/project/h3-turbo/)

**FluidGeo H3-Turbo** is a hardware-accelerated H3 spatial indexing library powered by SYCL (AdaptiveCpp). It provides high-performance, drop-in GPU/CPU-parallelized replacements for standard H3 operations, designed to operate seamlessly on NumPy arrays and PySpark DataFrames.

### Quick Links & Resources
* **[PyPI Project Page](https://pypi.org/project/h3-turbo/)**: View releases, installation requirements, and package details.
* **[h3_turbo_benchmarks.ipynb](file:///home/craig/dev/h3_turbo_v0.1/h3_turbo_benchmarks.ipynb)**: An interactive Jupyter notebook comparing GPU-accelerated operations against CPU-based `h3-py` and `numpy` equivalents across various data sizes (raw compute, spatial joins, batch processing, and GPU index caching reuse).
* **[spark_udf_tests.ipynb](file:///home/craig/dev/h3_turbo_v0.1/spark_udf_tests.ipynb)**: Interactive Jupyter notebook demonstrating PySpark integration, UDF usage, and persistent spatial join optimizations.

---

## Licensing

FluidGeo H3-Turbo is offered under a dual-license model:
* **Academic & Non-Commercial:** Free for research and educational purposes.
* **Commercial & Enterprise:** A yearly subscription is required for production environments.
  * *Features: Up to ~1000x speedup on Blackwell/Hopper GPUs, zero-copy pinned memory, multi-GPU scalability, and priority SYCL kernel support.*

For enterprise trial keys, support, and pricing, contact: **info@fluidgeollc.com**

---

## Installation

H3-Turbo is available on PyPI and comes with pre-compiled "fat" wheels for Linux (CUDA 12.x) supporting NVIDIA Ampere, Ada Lovelace, Hopper, and Blackwell architectures.

```bash
pip install <downloaded wheel path>
```

---

## Hardware Initialization & Verification

Before executing large workloads, verify that your SYCL acceleration backend is correctly recognized and warm up the JIT compiler.

```python
import h3_turbo

# 1. Check version (consistent with pyproject.toml)
print(f"H3 Turbo Version: {getattr(h3_turbo, '__version__', 'unknown')}")

# 2. Query active SYCL device
device = h3_turbo.device_name() # or h3_turbo.get_device_name()
print(f"Active Compute Device: {device}")

# 3. Warm up JIT compiler
h3_turbo.warmup()
print("JIT compilation warmed up and ready!")
```

---

## Python API Reference

H3-Turbo functions are optimized for high-throughput batch operations on NumPy arrays.

### 1. Lat/Lon to Cell Conversion
Convert coordinates (`lats`, `lons`) to H3 cell indexes at a specific resolution.
```python
import numpy as np

lats = np.random.uniform(37.7, 37.8, 1_000_000)
lngs = np.random.uniform(-122.5, -122.4, 1_000_000)
resolution = 9

# Returns a uint64 array of H3 indices
cells = h3_turbo.latlng_to_cell(lats, lngs, resolution)
```

### 2. Cell to Parent
Find the parent cells at a coarser resolution.
```python
parent_res = 5
parents = h3_turbo.cell_to_parent(cells, parent_res)
```

### 3. Grid Disk (k-ring)
Compute the grid disk of radius `k` around cell(s). Supports both scalar origins and batch arrays.
```python
# Scalar origin: returns 1D array of cells
single_disk = h3_turbo.grid_disk(0x8928308280fffff, k=2)

# Batch array: returns a 2D (N, max_k_size) array padded with 0s
disks = h3_turbo.grid_disk(cells, k=2)
```

### 4. Cell to Boundary
Get the lat/lng boundary coordinates of cells.
```python
# Returns an (N, 7, 2) array of [lat, lng] boundary vertices
boundaries = h3_turbo.cell_to_boundary(cells)

# Unoptimized 10-vertex boundary layout (for specific legacy compatibility)
boundaries_10 = h3_turbo.cell_to_boundary_10(cells)
```

### 5. Spatial Join (Point-in-Polygon / Inclusion Check)
Check if points (pings) are within a set of zones.
* **Production Overload:** Strictly 3 parameters, running at maximum GPU performance (no scramble).
* **Benchmarking Overload:** Includes the optional `scramble_iterations` parameter (e.g., set to `50` for matching baseline/benchmark scrambles).

```python
zones = np.array([0x8928308280fffff], dtype=np.uint64)

# 1. Production usage (no scramble_iterations needed)
mask = h3_turbo.spatial_join(cells, zones, resolution=9)

# 2. Benchmarking / verification usage
mask_bench = h3_turbo.spatial_join(cells, zones, resolution=9, scramble_iterations=50)
```

### 6. Persistent Joiner (Advanced Spatial Join)
For high-frequency point-in-polygon queries, avoid rebuilding the spatial index on every call by reusing a persistent instance. Use this when running many spatial join queries on the same set of zones.
```python
# 1. Production usage (no scramble_iterations needed)
joiner = h3_turbo.PersistentJoiner(zones, resolution=9)

# 2. Benchmarking / verification usage
joiner_bench = h3_turbo.PersistentJoiner(zones, resolution=9, scramble_iterations=50)

# Run multiple joins efficiently
results = np.zeros(len(cells), dtype=np.uint8)
joiner.join(cells, results)
```

### 7. Batch Transform
In-place GPU resolution transformation of an array of H3 indices.
```python
cells_to_transform = cells.copy()

# 1. Production usage (no scramble_iterations needed)
h3_turbo.batch_transform(cells_to_transform, res=8)

# 2. Benchmarking / verification usage
h3_turbo.batch_transform(cells_to_transform, res=8, scramble_iterations=50)
```

### 8. System Control & Cleanup
```python
# Set your enterprise license key to unlock full performance
h3_turbo.set_license_key("YOUR_LICENSE_KEY")

# Manually release internal GPU queue and SYCL resources
h3_turbo.cleanup()
```

---

## Spark / Databricks Integration

H3-Turbo provides high-throughput Pandas UDFs for PySpark, enabling distributed GPU execution.

```python
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Ensure workers run in the environment containing pyarrow and h3_turbo
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from spark_h3_turbo import (
    latlng_to_cell_udf,
    cell_to_parent_udf,
    grid_disk_udf,
    spatial_join_udf,
    persistent_spatial_join_udf,
    batch_transform_udf
)

spark = SparkSession.builder.appName("H3-Turbo-Spark").getOrCreate()

# 1. Lat/Lon to Cell
df = df.withColumn("h3", latlng_to_cell_udf(resolution=9)(col("lat"), col("lon")))

# 2. Cell to Parent
df = df.withColumn("parent", cell_to_parent_udf(parent_res=5)(col("h3")))

# 3. Grid Disk
df = df.withColumn("kring", grid_disk_udf(k=2)(col("h3")))

# 4. Spatial Join (Broadcast / Inclusion Check)
# Use spatial_join_udf for simple one-off queries
zones_list = [0x8928308280fffff]
df = df.withColumn("in_zone", spatial_join_udf(zones_list, res=9)(col("h3")))

# Use persistent_spatial_join_udf for large datasets. It caches the GPU spatial 
# index once per PySpark worker process and reuses it across all partition batches, 
# preventing index rebuild overhead.
df = df.withColumn("in_zone_persistent", persistent_spatial_join_udf(zones_list, res=9)(col("h3")))

# 5. Batch Transform
df = df.withColumn("transformed_h3", batch_transform_udf(res=8)(col("h3")))
```

