import pytest
import h3_turbo
import numpy as np
import h3
import time
from collections import Counter
import os

def numpy_apply_weight(h3_array):
    """
    Vectorized implementation of the 50-loop scramble using NumPy.
    """
    p = h3_array.astype(np.uint64)
    c1 = np.uint64(0xBF58476D1CE4E5B9)
    c2 = np.uint64(0x94D049BB133111EB)
    for _ in range(50):
        p ^= (p >> np.uint64(7))
        p *= c1
        p ^= (p >> np.uint64(13))
        p *= c2
        p ^= (p >> np.uint64(31))
    return p

@pytest.fixture(scope="session", autouse=True)
def warmup_gpu():
    print("Warming up the JIT...", flush=True)
    print(f"DEBUG: h3_turbo loaded from {h3_turbo.__file__}", flush=True)
    # Tiny dummy data to trigger compilation
    dummy_data = np.array([0x8928308280fffff], dtype=np.uint64)
    res = 5
    # Call the kernel to trigger compilation
    h3_turbo.batch_transform(dummy_data, res)
    print("🚀 JIT Ready!", flush=True)

def test_heatmap_performance():
    """
    Compares the performance of a heatmap generation on CPU vs. GPU.
    """
    n = int(os.environ.get("H3_NUM_PINGS", 10_000_000))
    res_target = 5
    base_index = 0x8928308280fffff
    
    print(f"\nGenerating {n} random H3 indexes...")
    # Use k_ring to get VALID indices, then sample from them.
    k = 200
    # H3 v4 uses strings. Convert int -> str -> grid_disk -> int
    valid_seeds = [h3.str_to_int(x) for x in h3.grid_disk(h3.int_to_str(base_index), k)]
    seeds_np = np.array(valid_seeds, dtype=np.uint64)
    pings = seeds_np[np.random.randint(0, len(seeds_np), size=n)]

    # Create a copy for CPU baseline because GPU kernel modifies in-place
    pings_cpu = pings.copy()

    # --- PATH A: GPU ACCELERATED ---
    print("Performing heatmap generation on GPU...")
    start_gpu = time.time()
    
    # The h3_turbo.batch_transform function is expected to perform the cellToParent and scramble
    gpu_results = h3_turbo.batch_transform(pings, res_target)
    
    gpu_duration = time.time() - start_gpu
    print(f"GPU heatmap took: {gpu_duration:.4f} seconds.")
    throughput = n / gpu_duration
    print(f"GPU Throughput: {throughput:,.0f} points/sec")

    # --- PATH B: PURE CPU ---
    print("\nPerforming heatmap generation on CPU...")
    start_cpu = time.time()

    # 1. Get parent for each H3 index
    cpu_parents = np.array([h3.str_to_int(h3.cell_to_parent(h3.int_to_str(h), res_target)) for h in pings_cpu], dtype=np.uint64)
    
    # 2. Apply the scramble to each parent
    cpu_results = numpy_apply_weight(cpu_parents)

    cpu_duration = time.time() - start_cpu
    print(f"CPU heatmap took: {cpu_duration:.4f} seconds.")

    # --- Verification and ROI ANALYSIS ---
    print("\n" + "=" * 60)
    print(f"DOORDASH WORKLOAD: ({n:,} Pings)")
    print(f"CPU Total Pipeline: {cpu_duration:.4f} seconds")
    print(f"GPU Total Pipeline: {gpu_duration:.4f} seconds")
    if gpu_duration > 0:
        print(f"OVERALL SPEEDUP:    {cpu_duration / gpu_duration:.2f}x")
    else:
        print("GPU execution was too fast to measure speedup.")
    print("=" * 60)
    
    # Verify that the results are the same
    print("\nVerifying results...")
    # Counter is too slow for 10M items. Use NumPy equality check.
    assert np.array_equal(gpu_results, cpu_results), "GPU and CPU results do not match!"
    print("✅ GPU and CPU results match.")

    print("\n---")
    print("Benchmark complete. To run the tests, use the command: pytest")
    print("---\n")

if __name__ == "__main__":
    if "H3_TURBO_LICENSE" in os.environ:
        h3_turbo.set_license_key(os.environ["H3_TURBO_LICENSE"].strip())
    # Manually trigger warmup since pytest fixtures don't run in script mode
    print("\n☕ Stan is warming up the JIT...", flush=True)
    dummy_data = np.array([0x8928308280fffff], dtype=np.uint64)
    h3_turbo.batch_transform(dummy_data, 5)
    print("🚀 JIT Ready!", flush=True)
    test_heatmap_performance()
