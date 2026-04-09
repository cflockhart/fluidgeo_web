import pytest
import h3_turbo
import numpy as np
import h3
import os

# Helper to handle h3-py v4 string/int conversion
def to_int(h3_idx):
    if isinstance(h3_idx, str):
        return h3.str_to_int(h3_idx)
    return h3_idx

def to_str(h3_idx):
    if isinstance(h3_idx, (int, np.integer, np.uint64)):
        return h3.int_to_str(h3_idx)
    return h3_idx

def test_cell_to_parent_correctness():
    """
    Verifies that GPU cell_to_parent matches CPU h3.cell_to_parent.
    """
    print("\nTesting cell_to_parent correctness...")
    n = 1000
    res = 9
    parent_res = 5
    
    # Generate random valid cells using a grid disk around SF
    base_lat, base_lng = 37.7749, -122.4194
    base_cell = h3.latlng_to_cell(base_lat, base_lng, res)
    
    # Get a neighborhood
    cells_set = h3.grid_disk(base_cell, 20)
    # Ensure we have enough
    cells_list = [to_int(c) for c in list(cells_set)[:n]]
    if len(cells_list) < n:
        print(f"Warning: Only generated {len(cells_list)} cells")
    
    cells_np = np.array(cells_list, dtype=np.uint64)
    
    # GPU Execution
    gpu_parents = h3_turbo.cell_to_parent(cells_np, parent_res)
    
    # CPU Execution
    cpu_parents = []
    for c in cells_list:
        p = h3.cell_to_parent(to_str(c), parent_res)
        cpu_parents.append(to_int(p))
    cpu_parents = np.array(cpu_parents, dtype=np.uint64)
    
    # Comparison
    mismatches = np.sum(gpu_parents != cpu_parents)
    assert mismatches == 0, f"Found {mismatches} mismatches in cell_to_parent"
    print("cell_to_parent passed.")

def test_grid_disk_correctness():
    """
    Verifies that GPU grid_disk matches CPU h3.grid_disk.
    """
    print("\nTesting grid_disk correctness...")
    n = 100
    k = 2
    res = 9
    
    base_lat, base_lng = 37.7749, -122.4194
    base_cell = h3.latlng_to_cell(base_lat, base_lng, res)
    
    cells_set = h3.grid_disk(base_cell, 5)
    cells_list = [to_int(c) for c in list(cells_set)[:n]]
    cells_np = np.array(cells_list, dtype=np.uint64)
    
    # GPU Execution
    # Returns (N, max_k_size)
    gpu_disks = h3_turbo.grid_disk(cells_np, k)
    
    # CPU Execution & Comparison
    for i, c in enumerate(cells_list):
        # CPU set
        expected_set = {to_int(x) for x in h3.grid_disk(to_str(c), k)}
        
        # GPU set (filter 0s)
        gpu_row = gpu_disks[i]
        gpu_set = {x for x in gpu_row if x != 0}
        
        # Check set equality
        assert gpu_set == expected_set, f"Mismatch for cell {to_str(c)}"
        
    print("grid_disk passed.")

def test_cell_to_boundary_correctness():
    """
    Verifies cell_to_boundary against standard H3.
    """
    print("\nTesting cell_to_boundary correctness...")
    n = 50
    res = 7
    
    base_lat, base_lng = 40.6892, -74.0445 # Statue of Liberty
    base_cell = h3.latlng_to_cell(base_lat, base_lng, res)
    
    cells_set = h3.grid_disk(base_cell, 5)
    cells_list = [to_int(c) for c in list(cells_set)[:n]]
    cells_np = np.array(cells_list, dtype=np.uint64)
    
    # GPU Execution
    # Returns (N, 10, 2)
    gpu_bounds = h3_turbo.cell_to_boundary(cells_np)
    
    for i, c in enumerate(cells_list):
        # CPU result: tuple of (lat, lng)
        cpu_bound = h3.cell_to_boundary(to_str(c))
        num_verts = len(cpu_bound)
        
        gpu_poly = gpu_bounds[i]
        
        # Check vertices
        for v in range(num_verts):
            lat_cpu, lng_cpu = cpu_bound[v]
            lat_gpu, lng_gpu = gpu_poly[v]
            
            # Use a reasonable tolerance for floating point differences
            assert np.isclose(lat_cpu, lat_gpu, atol=1e-7), f"Lat mismatch cell {i} v {v}: {lat_cpu} vs {lat_gpu}"
            assert np.isclose(lng_cpu, lng_gpu, atol=1e-7), f"Lng mismatch cell {i} v {v}: {lng_cpu} vs {lng_gpu}"
            
    print("cell_to_boundary passed.")

if __name__ == "__main__":
    # Setup license if running as script
    if "H3_TURBO_LICENSE" in os.environ:
        h3_turbo.set_license_key(os.environ["H3_TURBO_LICENSE"].strip())
    
    test_cell_to_parent_correctness()
    test_grid_disk_correctness()
    test_cell_to_boundary_correctness()