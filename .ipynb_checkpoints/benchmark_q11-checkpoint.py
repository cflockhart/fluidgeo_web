import time
import argparse
import numpy as np
import sys

# Try to import h3_turbo (the SYCL accelerated module)
try:
    import h3_turbo
except ImportError:
    print("Error: h3_turbo module not found. Please install the wheel or build locally.")
    sys.exit(1)

# Try to import standard h3 for baseline comparison
try:
    import h3
    HAS_H3_STD = True
except ImportError:
    HAS_H3_STD = False

def generate_trips(n_rows):
    """
    Generates synthetic trip data within NYC bounding box.
    Returns: (pickup_lats, pickup_lons, dropoff_lats, dropoff_lons)
    """
    # NYC Bounding Box
    lat_min, lat_max = 40.4774, 40.9176
    lon_min, lon_max = -74.2591, -73.7002
    
    # Use float64 for coordinates (standard for geospatial)
    # h3_turbo should handle this or cast internally
    p_lats = np.random.uniform(lat_min, lat_max, n_rows)
    p_lons = np.random.uniform(lon_min, lon_max, n_rows)
    d_lats = np.random.uniform(lat_min, lat_max, n_rows)
    d_lons = np.random.uniform(lon_min, lon_max, n_rows)
    
    return p_lats, p_lons, d_lats, d_lons

def run_benchmark(sizes, resolution=9):
    print(f"--- SpatiBench Q11 Approximation Benchmark (Resolution {resolution}) ---")
    print("Query: Count trips where pickup_cell != dropoff_cell")
    print(f"{'Rows':<12} | {'Method':<10} | {'Time (s)':<10} | {'Throughput (M/s)':<18} | {'Result':<10}")
    print("-" * 75)

    for n in sizes:
        # 1. Generate Data
        # print(f"Generating {n} rows...", file=sys.stderr)
        p_lats, p_lons, d_lats, d_lons = generate_trips(n)

        # 2. Benchmark h3_turbo
        # Warmup (optional, but good for JIT/GPU context initialization)
        if n == sizes[0]:
             _ = h3_turbo.latlng_to_cell(p_lats[:100], p_lons[:100], resolution)

        t0 = time.time()
        
        # Vectorized conversion
        p_cells = h3_turbo.latlng_to_cell(p_lats, p_lons, resolution)
        d_cells = h3_turbo.latlng_to_cell(d_lats, d_lons, resolution)
        
        # The "Join/Filter" step: Compare indices
        cross_zone_mask = p_cells != d_cells
        count_turbo = np.sum(cross_zone_mask)
        
        t1 = time.time()
        dt_turbo = t1 - t0
        throughput_turbo = (n / dt_turbo) / 1_000_000

        print(f"{n:<12} | {'h3_turbo':<10} | {dt_turbo:<10.4f} | {throughput_turbo:<18.2f} | {count_turbo}")

        # 3. Benchmark standard h3 (Baseline)
        # Only run for smaller sizes as it is very slow
        if HAS_H3_STD and n <= 1_000_000:
            t0 = time.time()
            
            # Standard h3 is scalar, requires loop
            # Using list comprehension which is generally fastest for scalar calls in Python
            p_cells_std = [h3.latlng_to_cell(lat, lon, resolution) for lat, lon in zip(p_lats, p_lons)]
            d_cells_std = [h3.latlng_to_cell(lat, lon, resolution) for lat, lon in zip(d_lats, d_lons)]
            
            # Convert to numpy for fast comparison (simulating best effort for standard lib)
            p_arr = np.array(p_cells_std)
            d_arr = np.array(d_cells_std)
            
            cross_zone_mask_std = p_arr != d_arr
            count_std = np.sum(cross_zone_mask_std)
            
            t1 = time.time()
            dt_std = t1 - t0
            throughput_std = (n / dt_std) / 1_000_000
            
            print(f"{n:<12} | {'h3 (std)':<10} | {dt_std:<10.4f} | {throughput_std:<18.2f} | {count_std}")
            print(f"   >>> Speedup: {dt_std / dt_turbo:.2f}x")
        elif HAS_H3_STD:
            print(f"{n:<12} | {'h3 (std)':<10} | {'SKIPPED':<10} | {'(Too slow)':<18} | -")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark SpatiBench Q11 Approximation")
    parser.add_argument('--sizes', type=int, nargs='+', default=[1_000_000, 10_000_000, 100_000_000],
                        help="List of dataset sizes to benchmark (default: 1M, 10M, 100M)")
    parser.add_argument('--res', type=int, default=9, help="H3 Resolution (default: 9)")
    
    args = parser.parse_args()
    
    run_benchmark(args.sizes, args.res)