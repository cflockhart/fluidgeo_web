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

# 6. Batch Transform
# In-place GPU transform of an array of H3 indices to a target resolution
cells_to_transform = cells.copy()
h3_turbo.batch_transform(cells_to_transform, 8)
```

## Spark / Databricks Integration

H3 Turbo includes optimized Pandas UDFs for PySpark.

```python
from pyspark.sql.functions import col
from spark_h3_turbo import (
    latlng_to_cell_udf,
    cell_to_parent_udf,
    grid_disk_udf,
    spatial_join_udf,
    batch_transform_udf
)

# 1. Lat/Lon to Cell
df = df.withColumn("h3", latlng_to_cell_udf(9)(col("lat"), col("lon")))

# 2. Cell to Parent
df = df.withColumn("parent", cell_to_parent_udf(5)(col("h3")))

# 3. Grid Disk
df = df.withColumn("kring", grid_disk_udf(2)(col("h3")))

# 4. Spatial Join (Broadcast)
zones_list = [0x8928308280fffff] # List of H3 integers
df = df.withColumn("in_zone", spatial_join_udf(zones_list, 9)(col("h3")))

# 5. Batch Transform
df = df.withColumn("transformed_h3", batch_transform_udf(8)(col("h3")))
```

## Building from Source

If you wish to build `h3-turbo` from source, either to contribute, customize, or target specific hardware configurations not covered by the pre-built wheels, follow these instructions.

### Prerequisites

Before you begin, ensure you have the following installed on your system:

*   **Git**: For cloning the repository.
*   **Python 3.10+**: With `pip` and `venv`.
*   **pipx**: For isolated installation of `cibuildwheel`.
    ```bash
    pip install pipx
    pipx ensurepath
    ```
*   **Docker**: With the `buildx` plugin enabled, for building multi-architecture Docker images.
*   **AdaptiveCpp (acpp)**: The SYCL compiler and runtime. Ensure `acpp` is in your system's `PATH`.
*   **CUDA Toolkit**: For NVIDIA GPUs. Ensure `nvcc` is in your `PATH`.
*   **ROCm**: For AMD GPUs. Ensure `hipcc` is in your `PATH`.
*   **H3 Library**: The H3 C library will be automatically built from source by `cibuildwheel` or `build_app.sh`.

### Building the Python Wheel

The `h3-turbo` Python wheel is built using `cibuildwheel`, which orchestrates builds for various Python versions and platforms. The `GPU_ARCH` environment variable is crucial for targeting specific GPU architectures.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/h3_turbo.git # Replace with actual repo URL
    cd h3_turbo
    ```

2.  **Clean previous build artifacts (recommended):**
    ```bash
    rm -rf build dist wheelhouse || true
    ```

3.  **Determine your host architecture:**
    ```bash
    set HOST_ARCH (uname -m)
    if test "$HOST_ARCH" = "aarch64" -o "$HOST_ARCH" = "arm64"
        set WHEEL_ARCH "aarch64"
    else
        set WHEEL_ARCH "x86_64"
    end
    echo "Detected host architecture for wheel: $WHEEL_ARCH"
    ```

4.  **Build the wheel:**
    You can specify the target `GPU_ARCH` for the wheel.
    *   **For a specific NVIDIA GPU architecture (e.g., Hopper `sm_90`):**
        ```bash
        set -x CIBW_ARCHS "$WHEEL_ARCH"; set -x CIBW_ENVIRONMENT "GPU_ARCH=sm_90"; pipx run cibuildwheel --platform linux
        ```
        This will produce a wheel named `h3_turbo_sm90-0.1.13+sm90-cp312-cp312-manylinux_*.whl`.
    *   **For a "fat" wheel (multiple NVIDIA architectures: `sm_86`, `sm_89`, `sm_90`):**
        ```bash
        set -x CIBW_ARCHS "$WHEEL_ARCH"; set -x CIBW_ENVIRONMENT "GPU_ARCH=fat"; pipx run cibuildwheel --platform linux
        ```
        This will produce a wheel named `h3_turbo-0.1.13-cp312-cp312-manylinux_*.whl`.
        *Note: `sm_100` (Blackwell) is automatically handled by targeting `sm_90` (Hopper) with PTX generation for forward compatibility.*
    *   **For AMD GPUs (e.g., `gfx90a`):**
        ```bash
        set -x CIBW_ARCHS "$WHEEL_ARCH"; set -x CIBW_ENVIRONMENT "GPU_ARCH=gfx90a"; pipx run cibuildwheel --platform linux
        ```
        This will produce a wheel named `h3_turbo_gfx90a-0.1.13+gfx90a-cp312-cp312-manylinux_*.whl`.

    The built wheel(s) will be located in the `wheelhouse/` directory.

### Building the Docker Image

The `build_docker.sh` script automates the process of building the Python wheel (if not already built) and then constructing the Docker image.

1.  **Ensure the wheel is built:**
    Run one of the `cibuildwheel` commands from the "Building the Python Wheel" section above. The `build_docker.sh` script will automatically find and copy the latest wheel from `wheelhouse/` to `dist/`.

2.  **Build the Docker image:**
    You can specify the target `GPU_ARCH` for the Docker image. This `GPU_ARCH` will be passed as a build argument to the Dockerfile and will influence the base image and CUDA/ROCm configurations.

    *   **Default (multi-arch NVIDIA: `sm_86,sm_89,sm_90,sm_100`):**
        ```bash
        ./build_docker.sh
        ```
        This will tag the image as `docker.io/cflockhart/h3_turbo_sm-86-sm-89-sm-90-sm-100:latest`.
    *   **Specific NVIDIA architecture (e.g., `sm_90`):**
        ```bash
        ./build_docker.sh sm_90
        ```
        This will tag the image as `docker.io/cflockhart/h3_turbo_sm-90:latest`.
    *   **AMD architecture (e.g., `gfx90a`):**
        ```bash
        ./build_docker.sh gfx90a
        ```
        This will tag the image as `docker.io/cflockhart/h3_turbo_gfx90a:latest`.
    *   **With Spark support:**
        ```bash
        ./build_docker.sh sm_90 --spark
        ```
        This will tag the image as `docker.io/cflockhart/h3_turbo_sm-90_spark:latest`.
    *   **Without pushing to Docker Hub:**
        ```bash
        ./build_docker.sh sm_90 --no-push
        ```

The script will handle building the wheel (if necessary), copying it to `dist/`, and then building the Docker image.
```
