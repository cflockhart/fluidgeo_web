import pytest
import h3_turbo
import numpy as np
import h3
import time
import os

def get_available_ram_gb():
    try:
        return (os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')) / (1024**3)
    except (ValueError, AttributeError):
        return 0

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

@pytest.mark.skipif(get_available_ram_gb() < 20, reason="Available RAM is less than 20GB")
def test_q11_spatial_join():
    """
    SpatialBench Query 11: Spatial Join (Point-in-Polygon).
    Measures performance of joining a large set of points (pings) against a set of polygons (zones).
    """
    n_pings = int(os.environ.get("H3_NUM_PINGS", 1_100_000_000))
    n_zones = 100_000_000
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
    # Generate in chunks to avoid OOM (peak memory reduction)
    pings = np.empty(n_pings, dtype=np.uint64)
    gen_chunk_size = 100_000_000
    for i in range(0, n_pings, gen_chunk_size):
        size = min(gen_chunk_size, n_pings - i)
        ix = np.random.randint(0, len(pool_np), size=size)
        pings[i : i + size] = pool_np[ix]
    
    print("Warming up JIT...")
    h3_turbo.warmup()

    # GPU Run
    print("Running GPU Spatial Join...")
    gpu_results = np.empty(n_pings, dtype=np.uint8)
    start_gpu = time.time()
    
    # Batch execution to prevent Host OOM during Python->C++ data transfer
    i = 0
    while i < n_pings:
        # Dynamic batch sizing based on available RAM
        try:
            # Get available RAM in bytes (Linux only)
            available_ram = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
        except (ValueError, AttributeError):
            # Fallback to a safe default (e.g., 2GB available) if detection fails
            available_ram = 2 * 1024 * 1024 * 1024

        # Heuristic: Use ~5% of available RAM per batch to be safe against spikes and fragmentation
        # Estimate 20 bytes per ping (8 input + 1 output + overhead)
        gpu_batch_size = int((available_ram * 0.05) / 20)
        # Clamp batch size to reasonable limits (1M to 100M)
        gpu_batch_size = max(1_000_000, min(1_000_000_000, gpu_batch_size))
        
        print(f"Dynamic GPU Batch Size: {gpu_batch_size:,} (Available RAM: {available_ram / 1024**3:.2f} GB)")
        
        end = min(i + gpu_batch_size, n_pings)
        gpu_results[i:end] = h3_turbo.spatial_join(pings[i:end], zones, res_target)
        i = end

    gpu_duration = time.time() - start_gpu
    print(f"GPU Time: {gpu_duration:.4f} s")

    # CPU Run
    print("Running CPU Spatial Join (Baseline)...")
    zone_parents = numpy_apply_weight(np.array([h3.str_to_int(h3.cell_to_parent(h3.int_to_str(z), res_target)) for z in zones], dtype=np.uint64))
    zone_set = set(zone_parents)

    start_cpu = time.time()
    cpu_results = np.zeros(n_pings, dtype=np.uint8)
    chunk_size = 1_000_000
    
    # Process in chunks to avoid OOM
    for i in range(0, n_pings, chunk_size):
        chunk = pings[i : i + chunk_size]
        chunk_parents = numpy_apply_weight(np.array([h3.str_to_int(h3.cell_to_parent(h3.int_to_str(p), res_target)) for p in chunk], dtype=np.uint64))
        cpu_results[i : i + chunk_size] = [1 if p in zone_set else 0 for p in chunk_parents]
        
    cpu_duration = time.time() - start_cpu
    print(f"CPU Time: {cpu_duration:.4f} s")

    print(f"\nSPEEDUP: {cpu_duration / gpu_duration:.2f}x")
    
    assert np.array_equal(gpu_results, cpu_results)
    print("Verification Passed.")

if __name__ == "__main__":
    if "H3_TURBO_LICENSE" in os.environ:
        h3_turbo.set_license_key(os.environ["H3_TURBO_LICENSE"].strip())
    test_q11_spatial_join()
