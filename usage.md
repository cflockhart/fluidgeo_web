# H3 Turbo Usage Guide

Accelerate your H3 geospatial indexing and spatial joins using SYCL-based GPU acceleration.

**H3 Turbo** Python bindings for high-performance H3 operations. Other language bindings coming soon, plus Windows, Mac and AMD support. offloading heavy computations to the GPU. It is designed for massive datasets where standard CPU-based H3 libraries become a bottleneck.

## Installation


# Install the package
Download the wheel file from the `dist` directory. It CUDA Toolkit >= 12 needs to be installed on an Ubuntu system 22.04+ with Python version required indicated in the file name: cp312 == python 3.12.
e.g.
```
pip install h3_turbo-0.0.1-cp312-cp312-manylinux_2_39_x86_64.whl
```
## Licensing

H3 Turbo requires a valid license key to operate. You must set the `H3_TURBO_LICENSE` environment variable or set it programmatically before calling API functions.

> Get a 1 month (or however long you need to evaluate it) by emailing info@fluidgeo.com

```python
import os
import h3_turbo

# Option 1: Set via Environment Variable (Recommended)
# export H3_TURBO_LICENSE="LICENSE-USER-EXP-SIGNATURE"

# Option 2: Set Programmatically
h3_turbo.set_license_key("YOUR_LICENSE_KEY_HERE")
```

## API Reference

The library exposes several key functions optimized for batch processing.

#### `latlng_to_cell(lats, lngs, res)`

Converts arrays of latitude/longitude pairs to H3 cell indices.

*   **lats**: Numpy array of latitudes.
*   **lngs**: Numpy array of longitudes.
*   **res**: H3 resolution (0-15).

#### `spatial_join(pings, zones, res_target)`

Performs a high-speed point-in-polygon (or point-in-set) check.

*   **pings**: Numpy array of H3 indices (the points to check).
*   **zones**: Numpy array of H3 indices (the "hot zones" or target areas).
*   **res_target**: The resolution at which to perform the containment check.

## Examples

### 1. Batch Point Indexing

```python
import h3_turbo
import numpy as np

# Generate dummy data
lats = np.random.uniform(37.7, 37.8, 1000000)
lngs = np.random.uniform(-122.5, -122.4, 1000000)
resolution = 9

# Run on GPU
cells = h3_turbo.latlng_to_cell(lats, lngs, resolution)

print(f"Indexed {len(cells)} points.")
```

### 2. Spatial Join (Geofencing)

```python
import h3_turbo
import numpy as np

# Load your data (e.g., from Parquet or CSV)
pings = np.load("pings.npy")  # Array of H3 indices
zones = np.load("zones.npy")  # Array of H3 indices (geofences)

# Perform the join at resolution 7
results = h3_turbo.spatial_join(pings, zones, 7)

# results[i] is 1 if pings[i] is inside any zone, 0 otherwise
matches = np.sum(results)
print(f"Found {matches} pings inside the zones.")
```

## Troubleshooting

#### Memory Issues (SIGKILL)

If you encounter a `SIGKILL` or crash during large benchmarks (e.g., >1 Billion points), it is likely due to running out of host RAM or GPU memory.

*   **Solution:** The library attempts to manage batch sizes dynamically. Ensure you have sufficient swap space or reduce the input size.