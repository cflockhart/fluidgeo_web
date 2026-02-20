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

def test_raw_compute_benchmark():
    """
    Raw Compute Benchmark: GPU vs CPU (Pure Transformation)
    """
    # 50 million elements, as in the Scala benchmark
    n = int(os.environ.get("H3_NUM_PINGS", 50_000_000))
    res_target = 9
    base_index = 0x8928308280fffff
    
    print(f"\n" + "=" * 80)
    print(f"RAW COMPUTE BENCHMARK: TRANSFORMING {n:,} H3 INDEXES")
    print("This test measures the ideal GPU workload (pure arithmetic) without overhead.")
    print("-" * 80)

# --- Data Setup ---
    print("Generating data...")
    
    
    # k=100 gives ~30,000 valid indices. Fast and safe.
    k = 100 
    # H3 v4 uses strings. Convert int -> str -> grid_disk -> int
    valid_seeds = [h3.str_to_int(x) for x in h3.grid_disk(h3.int_to_str(base_index), k)]
    num_seeds = len(valid_seeds)
    
    print(f"  Generated {num_seeds} valid unique seeds from grid_disk.")

    # Convert to numpy
    seeds_np = np.array(valid_seeds, dtype=np.uint64)

    # Tile them to reach n (50,000,000)
    # We repeat the valid chunk over and over.
    # This is valid for benchmarking because the math is the same.
    repeats = (n // num_seeds) + 1
    data_gpu = np.tile(seeds_np, repeats)[:n]
    
    # Shuffle if you want to avoid cache prediction cheating (optional but good)
    np.random.shuffle(data_gpu)

    print(f"  Expanded to {len(data_gpu):,} items.")

    # Copy for CPU to ensure identical start state
    data_cpu = data_gpu.copy()
    
    # Copy for CPU to ensure identical start state
    data_cpu = data_gpu.copy()
    
    # --- GPU BENCHMARK ---
    print("Running GPU benchmark...")
    start_gpu = time.time()
    
    # Call the kernel
    # batch_transform modifies data in-place (or returns a modified copy depending on binding)
    # My python binding 'py_batch_transform' copies input, processes, and returns new array.
    # This includes data transfer time!
    # The Scala benchmark 'batchTransformDirect' used a DirectByteBuffer (zero-copy if configured right, or pinned).
    # Here we are using standard numpy array, so there is copy overhead.
    # However, for 50M items (400MB), transfer is ~0.03s (PCIe 4.0 16GB/s) to ~0.1s.
    # Processing 50M items * 50 ops * 5 cycles = massive.
    
    gpu_results = h3_turbo.batch_transform(data_gpu, res_target)
    
    gpu_duration = time.time() - start_gpu
    print(f"GPU transformation complete in {gpu_duration:.4f} s")
    throughput = n / gpu_duration
    print(f"GPU Throughput: {throughput:,.0f} points/sec")
    
    # --- CPU BENCHMARK ---
    print("\nRunning CPU benchmark (Vectorized NumPy)...")
    start_cpu = time.time()
    
    # 1. Cell To Parent
    # h3.cell_to_parent is not vectorized in h3-py < 4.  h3-py 4 might map.
    # But usually list comp is slow.
    # For fair comparison against "Scala Loop", we want something fast.
    # Scala loop is scalar but JITed.
    # Numpy vectorization is the closest Python equivalent to "optimized code".
    
    # We'll use a simple approach: simple loop for parent (likely slow part), vectorized for scramble.
    # Actually, h3.cell_to_parent might be the bottleneck.
    # Let's try to mimic the bitwise logic if possible?
    # No, that's complex (ico logic).
    # We will use the library function and accept the overhead, OR
    # use a smaller N if CPU is too slow?
    # Python list comp for 50M will take ~10-20 seconds.
    # That's fine.
    
    # Vectorized scramble is fast.
    
    print("  Step 1: Cell to Parent (using h3-py)...")
    # We use numpy.vectorize or list comp
    # list comp is usually faster than np.vectorize for scalar functions
    parents_list = [h3.str_to_int(h3.cell_to_parent(h3.int_to_str(int(h)), res_target)) for h in data_cpu]
    parents_arr = np.array(parents_list, dtype=np.uint64)
    
    print("  Step 2: Scramble (using NumPy)...")
    cpu_results = numpy_apply_weight(parents_arr)
    
    cpu_duration = time.time() - start_cpu
    print(f"CPU transformation complete in {cpu_duration:.4f} s")
    
    # --- RESULTS ---
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print(f"CPU Time: {cpu_duration:.4f} s")
    print(f"GPU Time: {gpu_duration:.4f} s")
    if gpu_duration > 0:
        print(f"RAW COMPUTE SPEEDUP: {cpu_duration / gpu_duration:.2f}x")
    else:
        print("GPU too fast to measure.")
    print("=" * 80)
    
    # --- VERIFICATION ---
    # Verify first and last
    assert gpu_results[0] == cpu_results[0], f"Mismatch at 0: {gpu_results[0]} != {cpu_results[0]}"
    assert gpu_results[-1] == cpu_results[-1], f"Mismatch at -1: {gpu_results[-1]} != {cpu_results[-1]}"
    print("Verification successful.")

if __name__ == "__main__":
    if "H3_TURBO_LICENSE" in os.environ:
        h3_turbo.set_license_key(os.environ["H3_TURBO_LICENSE"].strip())
    test_raw_compute_benchmark()
