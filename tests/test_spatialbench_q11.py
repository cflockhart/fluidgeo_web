import pytest
import h3_turbo
import numpy as np
import h3
import time
import os
import concurrent.futures
import multiprocessing

def numpy_apply_weight(h3_array):
    """
    Vectorized implementation of the 50-loop scramble using NumPy.
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

# Global variable for worker processes to hold the zone set (avoids repeated pickling)
_global_zone_set = None

def _init_worker(zone_set):
    global _global_zone_set
    _global_zone_set = zone_set

def _process_chunk_cpu(chunk, res_target):
    # 1. Cell to parent (h3-py is scalar, so this loop is the bottleneck)
    parents = [h3.str_to_int(h3.cell_to_parent(h3.int_to_str(p), res_target)) for p in chunk]
    parents_np = np.array(parents, dtype=np.uint64)
    # 2. Scramble
    scrambled = numpy_apply_weight(parents_np)
    # 3. Check against global set
    return np.array([1 if p in _global_zone_set else 0 for p in scrambled], dtype=np.uint8)

def test_q11_spatial_join():
    """
    SpatialBench Query 11: Spatial Join (Point-in-Polygon).
    Measures performance of joining a large set of points (pings) against a set of polygons (zones).
    """
    n_pings = int(os.environ.get("H3_NUM_PINGS", 1_100_000_000))
    n_zones = int(os.environ.get("H3_NUM_ZONES", 1_000_000))
    res_target = 7
    base_index = 0x8928308280fffff
    
    print(f"\n" + "=" * 80)
    print(f"SPATIALBENCH QUERY 11: SPATIAL JOIN")
    print(f"Joining {n_pings:,} pings against {n_zones:,} zones")
    print("=" * 80)
    
    # Data Generation
    k = 200
    # H3 v4 uses strings. Convert int -> str -> grid_disk -> int
    pool = [h3.str_to_int(x) for x in h3.grid_disk(h3.int_to_str(base_index), k)]
    pool_np = np.array(pool, dtype=np.uint64)
    
    zones = np.random.choice(pool_np, n_zones, replace=(len(pool_np) < n_zones))
    
    print("Generating pings...")
    pings = pool_np[np.random.randint(0, len(pool_np), size=n_pings)]
    
    print("Warming up JIT...")
    h3_turbo.warmup()

    # GPU Run
    print("Running GPU Spatial Join...")
    start_gpu = time.time()
    # Encapsulated API: Pass zones directly
    gpu_results = h3_turbo.spatial_join(pings, zones, res_target)
    gpu_duration = time.time() - start_gpu
    print(f"GPU Time: {gpu_duration:.4f} s")

    if os.environ.get("SKIP_CPU") == "1":
        print("Skipping CPU verification.")
        return

    # CPU Run
    num_workers = multiprocessing.cpu_count()
    print(f"Running CPU Spatial Join (Baseline) on {num_workers} cores...")
    zone_parents = numpy_apply_weight(np.array([h3.str_to_int(h3.cell_to_parent(h3.int_to_str(z), res_target)) for z in zones], dtype=np.uint64))
    zone_set = set(zone_parents)

    start_cpu = time.time()
    chunk_size = 1_000_000
    chunks = [pings[i:i + chunk_size] for i in range(0, len(pings), chunk_size)]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker, initargs=(zone_set,)) as executor:
        cpu_results = np.concatenate(list(executor.map(_process_chunk_cpu, chunks, [res_target]*len(chunks))))
        
    cpu_duration = time.time() - start_cpu
    print(f"CPU Time: {cpu_duration:.4f} s")

    print(f"\nSPEEDUP: {cpu_duration / gpu_duration:.2f}x")
    
    assert np.array_equal(gpu_results, cpu_results)
    print("Verification Passed.")

if __name__ == "__main__":
    if "H3_TURBO_LICENSE" in os.environ:
        h3_turbo.set_license_key(os.environ["H3_TURBO_LICENSE"].strip())
    test_q11_spatial_join()
