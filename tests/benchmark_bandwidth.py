import h3_turbo
import numpy as np
import h3
import time
import os

def benchmark_bandwidth():
    print("=" * 80)
    print("H3-TURBO MEMORY BANDWIDTH BENCHMARK: cellToBoundary")
    print("Comparing Optimized (7 vertices) vs Original (10 vertices)")
    print("=" * 80)

    if "H3_TURBO_LICENSE" in os.environ:
        h3_turbo.set_license_key(os.environ["H3_TURBO_LICENSE"].strip())

    # Configuration
    N = 10_000_000
    base_index = 0x8928308280fffff

    print(f"Generating {N:,} cells...")
    # Generate valid cells
    k = 200
    pool = [h3.str_to_int(x) for x in h3.grid_disk(h3.int_to_str(base_index), k)]
    pool_np = np.array(pool, dtype=np.uint64)
    cells = pool_np[np.random.randint(0, len(pool_np), size=N)]

    # Warmup
    print("Warming up...")
    h3_turbo.cell_to_boundary(cells[:1000])
    h3_turbo.cell_to_boundary_10(cells[:1000])

    # --- Benchmark Optimized (7 verts) ---
    print("\n--- Optimized Kernel (7 vertices) ---")
    start = time.time()
    _ = h3_turbo.cell_to_boundary(cells)
    duration_7 = time.time() - start
    
    # Data Volume:
    # Input: N * 8 bytes
    # Output: N * 7 * 2 * 8 bytes
    bytes_7 = N * 8 + N * 7 * 2 * 8
    gb_7 = bytes_7 / 1e9
    bw_7 = gb_7 / duration_7
    
    print(f"Time: {duration_7:.4f} s")
    print(f"Data Volume: {gb_7:.2f} GB")
    print(f"Effective Bandwidth: {bw_7:.2f} GB/s")

    # --- Benchmark Original (10 verts) ---
    print("\n--- Original Kernel (10 vertices) ---")
    start = time.time()
    _ = h3_turbo.cell_to_boundary_10(cells)
    duration_10 = time.time() - start
    
    # Data Volume:
    # Input: N * 8 bytes
    # Output: N * 10 * 2 * 8 bytes
    bytes_10 = N * 8 + N * 10 * 2 * 8
    gb_10 = bytes_10 / 1e9
    bw_10 = gb_10 / duration_10
    
    print(f"Time: {duration_10:.4f} s")
    print(f"Data Volume: {gb_10:.2f} GB")
    print(f"Effective Bandwidth: {bw_10:.2f} GB/s")

    # --- Comparison ---
    print("\n" + "-" * 40)
    print(f"Speedup (Time): {duration_10 / duration_7:.2f}x")
    print(f"Memory Saving: {(bytes_10 - bytes_7) / 1e9:.2f} GB per run")
    print("-" * 40)

if __name__ == "__main__":
    benchmark_bandwidth()