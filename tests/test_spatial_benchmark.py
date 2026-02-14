import pytest
import h3_turbo
import numpy as np
import h3
import time
import os

def numpy_apply_weight(h3_array):
    """
    Vectorized implementation of the 50-loop scramble using NumPy.
    Matches the logic in RawPerformanceBenchmark.scala.
    """
    p = h3_array.astype(np.uint64)
    # Constants
    c1 = np.uint64(0xBF58476D1CE4E5B9)
    c2 = np.uint64(0x94D049BB133111EB)
    
    for _ in range(50):
        # p ^= (p >> 7)
        p ^= (p >> np.uint64(7))
        # p *= c1
        p *= c1
        # p ^= (p >> 13)
        p ^= (p >> np.uint64(13))
        # p *= c2
        p *= c2
        # p ^= (p >> 31)
        p ^= (p >> np.uint64(31))
        
    return p

def test_spatial_inclusion_performance():
    """
    Benchmarks checking if points are inside a set of polygons (Hot Zones).
    """
    n_pings = 10_000_000
    n_zones = 100_000 # Number of hot zone cells
    res_target = 7
    base_index = 0x8928308280fffff
    
    print(f"\nGenerating {n_pings} pings and {n_zones} hot zones...")
    
    # FIX: Use grid_disk to generate VALID H3 indices
    # k=200 gives ~120,000 valid indices.
    k = 200
    # H3 v4 uses strings. Convert int -> str -> grid_disk -> int
    pool = [h3.str_to_int(x) for x in h3.grid_disk(h3.int_to_str(base_index), k)]
    pool_np = np.array(pool, dtype=np.uint64)
    
    # Generate zones (subset of pool)
    zones = np.random.choice(pool_np, n_zones, replace=(len(pool_np) < n_zones))

    # Generate pings (random draw from pool)
    pings = pool_np[np.random.randint(0, len(pool_np), size=n_pings)]
    
    # --- GPU BENCHMARK ---
    print("\nPerforming spatial inclusion check on GPU...")
    start_gpu = time.time()
    
    # Run the kernel
    gpu_results = h3_turbo.spatial_join(pings, zones, res_target)
    
    gpu_duration = time.time() - start_gpu
    print(f"GPU spatial join took: {gpu_duration:.4f} seconds.")
    
    # --- CPU BENCHMARK ---
    print("\nPerforming spatial inclusion check on CPU...")
    # For CPU, we use a Python set for O(1) lookup, which is idiomatic and fast in Python
    # But to be fair, we should include the parent conversion time as the GPU does it.
    
    # Pre-calculate set to be fair (GPU table build is excluded from GPU timer)
    zone_parents_unscrambled = np.array([h3.str_to_int(h3.cell_to_parent(h3.int_to_str(z), res_target)) for z in zones], dtype=np.uint64)
    zone_parents_scrambled = numpy_apply_weight(zone_parents_unscrambled)
    zone_set = set(zone_parents_scrambled)

    start_cpu = time.time()
    
    # 1. Convert pings to parent
    cpu_parents = np.array([h3.str_to_int(h3.cell_to_parent(h3.int_to_str(p), res_target)) for p in pings], dtype=np.uint64)
    
    # 2. Scramble (Fair comparison)
    cpu_parents = numpy_apply_weight(cpu_parents)

    # 3. Check against set
    cpu_results = np.array([1 if p in zone_set else 0 for p in cpu_parents], dtype=np.uint8)
    
    cpu_duration = time.time() - start_cpu
    print(f"CPU spatial join took: {cpu_duration:.4f} seconds.")
    
    # --- COMPARISON ---
    print("\n" + "=" * 60)
    print(f"SPATIAL JOIN: ({n_pings:,} Pings against {n_zones:,} Zones)")
    print(f"CPU Time: {cpu_duration:.4f} s")
    print(f"GPU Time: {gpu_duration:.4f} s")
    if gpu_duration > 0:
        print(f"SPEEDUP:  {cpu_duration / gpu_duration:.2f}x")
    print("=" * 60)
    
    # Verify results
    matches = np.sum(gpu_results)
    print(f"Found {matches} matches.")
    assert np.array_equal(gpu_results, cpu_results), "GPU and CPU results do not match!"

if __name__ == "__main__":
    if "H3_TURBO_LICENSE" in os.environ:
        h3_turbo.set_license_key(os.environ["H3_TURBO_LICENSE"].strip())
    test_spatial_inclusion_performance()