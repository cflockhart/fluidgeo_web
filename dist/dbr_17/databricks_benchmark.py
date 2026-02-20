# Databricks notebook source
# MAGIC %md
# MAGIC # H3 Turbo vs Databricks SQL Benchmark
# MAGIC
# MAGIC This notebook benchmarks the performance of `h3-turbo` (GPU-accelerated) against standard Databricks SQL / Spark (CPU) for H3 geospatial operations.
# MAGIC
# MAGIC **Benchmarks:**
# MAGIC 1.  **Raw Compute**: `cell_to_parent` followed by a heavy bitwise "scramble" operation (simulating complex hashing/encryption).
# MAGIC 2.  **Spatial Join**: Point-in-Polygon check (Query 11 from SpatialBench).
# MAGIC
# MAGIC **Environment Setup:**
# MAGIC *   **Cluster**: Single User Cluster with GPU (e.g., AWS `g4dn.xlarge` or `g5.xlarge`).
# MAGIC *   **Runtime**: Databricks Runtime 14.3 LTS ML or higher (includes H3 functions).
# MAGIC *   **Libraries**: Ensure `h3` and `h3-turbo` (wheel) are installed.

# COMMAND ----------

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h3
from pyspark.sql.functions import col, pandas_udf, lit, expr
from pyspark.sql.types import LongType

# Try to import h3_turbo
try:
    import h3_turbo
    print(f"H3 Turbo Version: {h3_turbo.__file__}")
except ImportError:
    print("WARNING: h3_turbo not found. Please install the wheel via 'Compute > Libraries'.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration & Data Generation
# MAGIC We generate a synthetic dataset of **50 Million** H3 indices.

# COMMAND ----------

# Benchmark Settings
N_PINGS = 50_000_000  # 50 Million points
N_ZONES = 100_000     # 100k Polygons (Hot Zones)
RES_RAW = 9           # Resolution for Raw Benchmark
RES_JOIN = 7          # Resolution for Spatial Join

# Set License Key if available
if "H3_TURBO_LICENSE" in os.environ:
    h3_turbo.set_license_key(os.environ["H3_TURBO_LICENSE"])
else:
    print("Error: H3_TURBO_LICENSE environment variable is not set. Exiting.")
    exit(1)

def generate_data(n_pings, n_zones):
    print(f"Generating {n_pings:,} pings and {n_zones:,} zones...")
    # Use a valid base cell to generate real neighbors
    base_index = 0x8928308280fffff
    k = 200
    
    # Generate pool of valid indices using k-ring (V3 API)
    # H3 v3 uses string inputs/outputs for most functions
    pool = [h3.string_to_h3(x) for x in h3.k_ring(h3.h3_to_string(base_index), k)]
    pool_np = np.array(pool, dtype=np.uint64)
    
    # Sample from pool
    zones = np.random.choice(pool_np, n_zones, replace=(len(pool_np) < n_zones))
    pings = pool_np[np.random.randint(0, len(pool_np), size=n_pings)]
    
    return pings, zones

# Generate Data (Driver / Local)
pings_np, zones_np = generate_data(N_PINGS, N_ZONES)

# Create Spark DataFrames for comparison
print("Creating Spark DataFrames...")
# Note: Spark uses signed int64. H3 indices are 63-bit positive, so conversion is safe.
pings_df = spark.createDataFrame(pd.DataFrame({"cell": pings_np.astype(np.int64)}))
zones_df = spark.createDataFrame(pd.DataFrame({"cell": zones_np.astype(np.int64)}))

# Force materialization / cache to ensure fair timing (exclude data gen time)
pings_df.cache().count()
zones_df.cache().count()
print("DataFrames cached and ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Benchmark: Raw Compute (Transform)
# MAGIC **Logic**: `h3_to_parent(cell, 9)` -> `scramble(parent)`
# MAGIC
# MAGIC *   **Spark**: Uses native `h3_toparent` (highly optimized C++) + a **Vectorized Pandas UDF** for the scramble logic. This represents a "best practice" optimized Python workflow on Spark.
# MAGIC *   **H3 Turbo**: Uses `batch_transform` which runs a fused CUDA kernel performing both operations.

# COMMAND ----------

# Define Scramble Logic (NumPy / Pandas UDF)
def numpy_apply_weight(h3_array):
    """
    Vectorized implementation of the 50-loop scramble using NumPy.
    
    Purpose:
    Simulates a heavy compute workload (hashing) to benchmark CPU vs GPU integer throughput.
    Based on the SplitMix64 algorithm, repeated 50 times to increase arithmetic intensity.

    Constants:
    - c1 (0xBF58476D1CE4E5B9) & c2 (0x94D049BB133111EB): SplitMix64 constants used for mixing.

    Bitwise Operations:
    - XOR Shifts (>> 7, 13, 31): Mixes bits from higher positions into lower positions (Avalanche).
    - Multiplications (* c1, * c2): Non-linear mixing operations that spread entropy.
    """
    # Cast to uint64 for correct bitwise operations
    p = h3_array.astype(np.uint64)
    c1 = np.uint64(0xBF58476D1CE4E5B9)
    c2 = np.uint64(0x94D049BB133111EB)
    for _ in range(50):
        p ^= (p >> np.uint64(7))
        p *= c1
        p ^= (p >> np.uint64(13))
        p *= c2
        p ^= (p >> np.uint64(31))
    return p.astype(np.int64) # Return as int64 for Spark

@pandas_udf(LongType())
def scramble_udf(batch: pd.Series) -> pd.Series:
    return pd.Series(numpy_apply_weight(batch.values))

print(f"--- Running Spark SQL Benchmark (Raw Transform {N_PINGS:,} rows) ---")
start_spark = time.time()

# Spark Execution: Native H3 -> Pandas UDF -> Noop Write (Trigger)
(pings_df
 .withColumn("parent", expr(f"h3_toparent(cell, {RES_RAW})"))
 .withColumn("scrambled", scramble_udf(col("parent")))
 .write.format("noop").mode("overwrite").save())

spark_duration = time.time() - start_spark
raw_spark_time = spark_duration
print(f"Spark SQL Time: {spark_duration:.4f} s")

# --- H3 Turbo ---
print(f"--- Running H3 Turbo Benchmark (Raw Transform {N_PINGS:,} rows) ---")
h3_turbo.warmup() # Ensure JIT is ready

start_gpu = time.time()
gpu_results = h3_turbo.batch_transform(pings_np, RES_RAW)
gpu_duration = time.time() - start_gpu
raw_gpu_time = gpu_duration

print(f"H3 Turbo (GPU) Time: {gpu_duration:.4f} s")
print(f"Speedup: {spark_duration / gpu_duration:.2f}x")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Benchmark: Spatial Join (Query 11)
# MAGIC **Logic**: Find which pings fall inside any of the hot zones.
# MAGIC
# MAGIC *   **Spark**: `INNER JOIN` on `h3_toparent`. Spark optimizes this using Broadcast Hash Join (if zones are small) or Sort Merge Join.
# MAGIC *   **H3 Turbo**: `spatial_join` kernel.

# COMMAND ----------

print(f"--- Running Spark SQL Benchmark (Spatial Join) ---")
start_spark = time.time()

# Spark Execution: Calculate Parents -> Join -> Count
pings_parents = pings_df.withColumn("parent", expr(f"h3_toparent(cell, {RES_JOIN})"))
zones_parents = zones_df.withColumn("parent", expr(f"h3_toparent(cell, {RES_JOIN})"))

matches_spark = pings_parents.join(zones_parents, "parent", "inner").count()

spark_duration = time.time() - start_spark
join_spark_time = spark_duration
print(f"Spark SQL Time: {spark_duration:.4f} s")
print(f"Matches Found: {matches_spark}")

# --- H3 Turbo ---
print(f"--- Running H3 Turbo Benchmark (Spatial Join) ---")
start_gpu = time.time()

# H3 Turbo Execution: Returns boolean array (1 if match, 0 if not)
gpu_results = h3_turbo.spatial_join(pings_np, zones_np, RES_JOIN)
matches_gpu = np.sum(gpu_results)

gpu_duration = time.time() - start_gpu
join_gpu_time = gpu_duration
print(f"H3 Turbo (GPU) Time: {gpu_duration:.4f} s")
print(f"Matches Found: {matches_gpu}")

print(f"Speedup: {spark_duration / gpu_duration:.2f}x")

# Verification
assert matches_spark == matches_gpu, f"Mismatch! Spark: {matches_spark}, GPU: {matches_gpu}"
print("✅ Results Match")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Visualization
# MAGIC Visualizing the performance gap and the spatial data distribution.

# COMMAND ----------

# 1. Performance Comparison (Log Scale)
tasks = ['Raw Transform', 'Spatial Join']
cpu_times = [raw_spark_time, join_spark_time]
gpu_times = [raw_gpu_time, join_gpu_time]

x = np.arange(len(tasks))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Execution Time
rects1 = ax1.bar(x - width/2, cpu_times, width, label='Spark (CPU)', color='#FF9999')
rects2 = ax1.bar(x + width/2, gpu_times, width, label='H3 Turbo (GPU)', color='#66B2FF')

ax1.set_ylabel('Time (seconds) - Log Scale')
ax1.set_title('Execution Time (Lower is Better)')
ax1.set_xticks(x)
ax1.set_xticklabels(tasks)
ax1.legend()
ax1.set_yscale('log') # Log scale because GPU is likely orders of magnitude faster
ax1.grid(True, which="both", ls="-", alpha=0.2)

# Plot 2: Speedup Factor
speedups = [c / g for c, g in zip(cpu_times, gpu_times)]
bars = ax2.bar(tasks, speedups, color='green', alpha=0.7)
ax2.set_ylabel('Speedup Factor (X times faster)')
ax2.set_title('GPU Acceleration Factor')
ax2.bar_label(bars, fmt='%.1fx', padding=3)

plt.tight_layout()
plt.savefig('benchmark_performance.png')
plt.show()

# 2. Spatial Distribution Sample
# Plotting 50M points is impossible, so we sample 10k
sample_size = 10_000
print(f"Plotting spatial sample of {sample_size} points...")

pings_sample = np.random.choice(pings_np, sample_size, replace=False)
zones_sample = np.random.choice(zones_np, sample_size, replace=False)

# Convert H3 to Lat/Lon for plotting
pings_coords = np.array([h3.h3_to_geo(h3.int_to_str(h)) for h in pings_sample])
zones_coords = np.array([h3.h3_to_geo(h3.int_to_str(h)) for h in zones_sample])

plt.figure(figsize=(10, 10))
plt.scatter(pings_coords[:, 1], pings_coords[:, 0], c='blue', alpha=0.1, s=1, label='Pings')
plt.scatter(zones_coords[:, 1], zones_coords[:, 0], c='red', alpha=0.5, s=5, label='Hot Zones')
plt.title(f'Spatial Distribution (Sample of {sample_size})')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.savefig('spatial_distribution.png')
plt.show()