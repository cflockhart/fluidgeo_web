# H3 SYCL Bridge

## ⚖️ Licensing
FluidGeo H3-Turbo is offered under a dual-license model:
* **Academic & Non-Commercial:** Free for research and educational purposes.
* **Commercial & Enterprise:** A yearly subscription is required for production environments. 
  * *Features: 1,186x speedup on Blackwell, zero-copy pinned memory, and priority SYCL kernel support.*

For enterprise trial keys and pricing, contact: **info@fluidgeollc.com**

## Installation

H3-Turbo is available on PyPI and comes with pre-compiled "fat" wheels for Linux (CUDA 12.x) supporting NVIDIA Ampere, Ada Lovelace, Hopper, and Blackwell architectures.

```bash
pip install h3-turbo
```

## Python API

H3 Turbo provides drop-in replacements for common H3 functions, optimized for NumPy arrays and GPU acceleration.

```python
import h3_turbo
import numpy as np

# 1. Lat/Lon to Cell
lats = np.random.uniform(37.7, 37.8, 1_000_000)
lngs = np.random.uniform(-122.5, -122.4, 1_000_000)
resolution = 9

# Returns uint64 array of H3 indices
cells = h3_turbo.latlng_to_cell(lats, lngs, resolution)

# 2. Cell to Parent
parent_res = 5
parents = h3_turbo.cell_to_parent(cells, parent_res)

# 3. Grid Disk (k-ring)
k = 2
# Returns (N, max_k_size) array, padded with 0s
disks = h3_turbo.grid_disk(cells, k)

# 4. Cell to Boundary
# Returns (N, 7, 2) array of [lat, lng] coordinates
boundaries = h3_turbo.cell_to_boundary(cells)

# 5. Spatial Join (Point-in-Polygon)
# Efficiently check if points are within a set of zones
zones = np.array([0x8928308280fffff], dtype=np.uint64)
mask = h3_turbo.spatial_join(cells, zones, resolution)
```

## Spark / Databricks Integration

H3 Turbo includes optimized Pandas UDFs for PySpark.

```python
from pyspark.sql.functions import col
from spark_h3_turbo import (
    latlons_to_h3s_udf,
    cell_to_parent_udf,
    grid_disk_udf,
    spatial_join_udf
)

# 1. Lat/Lon to Cell
df = df.withColumn("h3", latlons_to_h3s_udf(9)(col("lat"), col("lon")))

# 2. Cell to Parent
df = df.withColumn("parent", cell_to_parent_udf(5)(col("h3")))

# 3. Grid Disk
df = df.withColumn("kring", grid_disk_udf(2)(col("h3")))

# 4. Spatial Join (Broadcast)
zones_list = [0x8928308280fffff] # List of H3 integers
df = df.withColumn("in_zone", spatial_join_udf(zones_list, 9)(col("h3")))
```
When choosing a wheel file or Docker image for AWS, refer to the following table:

| AWS Instance | GPU | Architecture | GPU_ARCH |
| :--- | :--- | :--- | :--- |
| `g5` | NVIDIA A10G | Ampere | `sm_86` |
| `p4d` | NVIDIA A100 | Ampere | `sm_80` |
| `g6` | NVIDIA L4 | Ada Lovelace | `sm_89` |
| `g6e` | NVIDIA L40S | Ada Lovelace | `sm_89` |
| `p5` | NVIDIA H100 | Hopper | `sm_90` |
| `p5e` | NVIDIA H200 | Hopper | `sm_90` |
| `g7e` | NVIDIA B200 | Blackwell | `sm_100` |
