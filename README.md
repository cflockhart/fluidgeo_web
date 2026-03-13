# H3 Turbo

## Python API

H3 Turbo provides drop-in replacements for common H3 functions, optimized for NumPy arrays and GPU acceleration.

## Benchmark files and example usage for h3_turbo

Wheel files containing the Python bindings and benchmarnk notebooks are in the `dist` directory.
Docker images for Nvidia CUDA compute capability 8.9, 9.0, 10.0  and 12.0 with Jupyter lab are available at `docker.io/cflockhart/h3_turbo_sm_89:latest`, `docker.io/cflockhart/h3_turbo_sm_90:latest` etc.

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
| `g4dn` | NVIDIA T4 | Turing | `sm_75` |
| `g5` | NVIDIA A10G | Ampere | `sm_86` |
| `p4d` | NVIDIA A100 | Ampere | `sm_80` |
| `g6` | NVIDIA L4 | Ada Lovelace | `sm_89` |
| `g6e` | NVIDIA L40S | Ada Lovelace | `sm_89` |
| `p5` | NVIDIA H100 | Hopper | `sm_90` |
| `p5e` | NVIDIA H200 | Hopper | `sm_90` |
| `g7e` | NVIDIA B200 | Blackwell | `sm_100` |

## Docker Prerequisites

To run the Docker images with GPU acceleration enabled (`h3-turbo`), you must ensure your host machine is correctly configured with NVIDIA drivers and Docker support.

Specifically, the following must be installed:

1.  **NVIDIA Drivers**: Ensure you have the NVIDIA GPU drivers installed on your host (compatible with CUDA 12.0+).
2.  **nvidia-container-toolkit**: This toolkit enables the Docker engine to access the GPU.

### Installation Guide

For the **NVIDIA Container Toolkit**, please follow the official installation guide.

After installing the toolkit, remember to restart the Docker daemon:
```bash
sudo systemctl restart docker
```

You can then verify your setup by running:
```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

## Running Automated Benchmarks via Docker

To run the automated benchmarks using Docker, you can use the provided `docker-compose.benchmark.yml` file. This setup automatically builds the necessary environment and executes `benchmark_runner.py` with GPU support enabled.

Make sure you have your `H3_TURBO_LICENSE` environment variable set, or pass it directly. Run the following command:

```bash
H3_TURBO_LICENSE=your_license_here docker compose -f docker-compose.benchmark.yml up --build
```

**Note:** Generated 1-month licenses are available within the Docker images published at [https://hub.docker.com/repositories/cflockhart](https://hub.docker.com/repositories/cflockhart).