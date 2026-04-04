import os
import sys
import time
import numpy as np
import h3
import json
import random
from datetime import datetime, timedelta

# Add project root and build directory to path to allow importing h3_turbo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))

try:
    import h3_turbo
except ImportError:
    print("Error: h3_turbo module not found. Please build the project first.")
    sys.exit(1)

try:
    import folium
    from folium import plugins
    import branca.colormap as cm
except ImportError:
    print("Error: folium is required for this demo. Please install it: pip install folium branca")
    sys.exit(1)

def numpy_apply_weight(h3_array):
    """
    Vectorized implementation of the scramble using NumPy (CPU Baseline).
    Matches the logic in test_benchmark.py.
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

def generate_temporal_la_data(res=8):
    """
    Generates synthetic temporal data (16:00 - 20:00) for Los Angeles.
    Returns:
        flat_pings: Array of H3 indices for benchmarking (simulating total load).
        geojson_features: List of GeoJSON features for the animation.
    """
    print(f"Generating synthetic temporal data (16:00 - 20:00)...")
    
    # LA Center
    lat, lng = 34.0522, -118.2437
    
    # Use h3-py v4 API
    center_h3 = h3.latlng_to_cell(lat, lng, res)
    
    # Get a disk of valid cells (k=15 at res 8 covers central LA)
    valid_cells = list(h3.grid_disk(center_h3, 15))
    
    zones = []
    for cell in valid_cells:
        pop = np.random.randint(1000, 50000) # Population
        wealth = np.random.randint(1, 11)    # Wealth index 1-10
        # Base rate factor
        zones.append({
            "h3": cell,
            "pop": pop,
            "wealth": wealth,
            "base_rate": (pop * wealth) / 500_000.0 
        })
        
    # Time settings
    start_dt = datetime(2024, 1, 1, 16, 0, 0)
    end_dt = datetime(2024, 1, 1, 20, 0, 0)
    delta = timedelta(minutes=10)
    
    current_dt = start_dt
    all_pings = []
    geojson_features = []
    
    while current_dt <= end_dt:
        time_str = current_dt.isoformat()
        
        # Time factor: Peak at 18:30
        hour_offset = (current_dt - start_dt).total_seconds() / 3600.0
        # Curve peaking at 2.5 (18:30)
        time_factor = 1.0 - 0.5 * abs(hour_offset - 2.5) / 2.5
        time_factor = max(0.2, time_factor)
        
        for z in zones:
            # Expected orders in this 10 min window (600 seconds)
            # Multiplier 200 to get enough volume for benchmark
            expected = int(z["base_rate"] * time_factor * 600 * 200)
            count = np.random.poisson(expected) if expected > 0 else 0
            
            if count > 0:
                # Add to flat list for benchmark
                batch = np.full(count, h3.str_to_int(z["h3"]), dtype=np.uint64)
                all_pings.append(batch)
            
            orders_per_sec = count / 600.0
            
            # Feature for animation
            boundary = h3.cell_to_boundary(z["h3"])
            poly_coords = [[(lng, lat) for lat, lng in boundary]]
            poly_coords[0].append(poly_coords[0][0])
            
            feature = {
                'type': 'Feature',
                'geometry': {'type': 'Polygon', 'coordinates': poly_coords},
                'properties': {
                    'time': time_str,
                    'style': {'fillOpacity': 0.6, 'weight': 0},
                    'orders_per_sec': orders_per_sec,
                    'popup': f"<b>Zone:</b> {z['h3']}<br><b>Time:</b> {current_dt.strftime('%H:%M')}<br><b>Pop:</b> {z['pop']:,}<br><b>Wealth:</b> {z['wealth']}/10<br><b>Orders/s:</b> {orders_per_sec:.2f}"
                }
            }
            geojson_features.append(feature)
            
        current_dt += delta
        
    flat_pings = np.concatenate(all_pings)
    np.random.shuffle(flat_pings)
    
    return flat_pings, geojson_features

def run_demo():
    # Configuration
    RES_TARGET = 8       # Aggregate to neighborhoods (Resolution 8)
    
    # 1. Setup Data
    pings_gpu, geojson_features = generate_temporal_la_data(RES_TARGET)
    N_PINGS = len(pings_gpu)
    pings_cpu = pings_gpu.copy()
    
    # 2. Warmup GPU
    print("Warming up GPU JIT...")
    dummy = np.array([pings_gpu[0]], dtype=np.uint64)
    h3_turbo.batch_transform(dummy, RES_TARGET)
    
    # 3. GPU Benchmark
    print(f"Running GPU Batch Transform ({N_PINGS:,} items)...")
    start_gpu = time.time()
    gpu_results = h3_turbo.batch_transform(pings_gpu, RES_TARGET)
    gpu_time = time.time() - start_gpu
    print(f"GPU Time: {gpu_time:.4f} s")
    
    # 4. CPU Benchmark
    print(f"Running CPU Baseline ({N_PINGS:,} items)...")
    start_cpu = time.time()
    # CPU Step 1: Cell to Parent (Standard h3-py approach)
    parents = np.array([h3.str_to_int(h3.cell_to_parent(h3.int_to_str(p), RES_TARGET)) for p in pings_cpu], dtype=np.uint64)
    # CPU Step 2: Scramble
    cpu_results = numpy_apply_weight(parents)
    cpu_time = time.time() - start_cpu
    print(f"CPU Time: {cpu_time:.4f} s")
    
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    print(f"Speedup: {speedup:.2f}x")
    
    # 5. Visualization
    print("Generating Animated Heatmap HTML...")
    
    # Create Map (Dark theme for "Cyberpunk/Tech" feel)
    m = folium.Map(location=[34.0522, -118.2437], zoom_start=11, tiles="CartoDB dark_matter")
    
    # Color scale based on max orders/sec in the simulation
    max_ops = max(f['properties']['orders_per_sec'] for f in geojson_features)
    colormap = cm.LinearColormap(
        colors=['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000'],
        vmin=0, vmax=max_ops,
        caption='Orders Per Second'
    )
    m.add_child(colormap)
    
    # Apply colors to features
    for f in geojson_features:
        ops = f['properties']['orders_per_sec']
        f['properties']['style']['fillColor'] = colormap(ops)
    
    # Add TimestampedGeoJson
    plugins.TimestampedGeoJson(
        {'type': 'FeatureCollection', 'features': geojson_features},
        period='PT10M',
        add_last_point=False,
        auto_play=True,
        loop=True,
        max_speed=10,
        loop_button=True,
        date_options='HH:mm',
        time_slider_drag_update=True
    ).add_to(m)

    # Add Performance Overlay with CSS Animation
    html_overlay = f"""
    <style>
        @keyframes pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(0, 210, 255, 0.4); }}
            70% {{ box-shadow: 0 0 0 10px rgba(0, 210, 255, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(0, 210, 255, 0); }}
        }}
        .stats-box {{
            animation: pulse 2s infinite;
            position: fixed; 
            top: 20px; right: 20px; width: 300px; height: 170px; 
            background-color: rgba(0,0,0,0.85); color: white; z-index:9999; 
            padding: 15px; border-radius: 10px; font-family: sans-serif; border: 1px solid #444;
        }}
    </style>
    <div class="stats-box">
        <h3 style="margin-top:0; color: #00d2ff; text-align: center;">H3 Turbo Benchmark</h3>
        <div style="margin-bottom: 5px; font-size: 0.9em;"><b>Scenario:</b> 4-Hour Traffic (16:00-20:00)</div>
        <div style="margin-bottom: 5px; font-size: 0.9em;"><b>Points:</b> {N_PINGS:,}</div>
        <hr style="border-color: #555;">
        <div style="display: flex; justify-content: space-between;">
            <span>CPU Time:</span> <span style="color: #ff6b6b;">{cpu_time:.4f} s</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>GPU Time:</span> <span style="color: #51cf66;">{gpu_time:.4f} s</span>
        </div>
        <div style="margin-top: 15px; font-size: 1.4em; text-align: center; color: #fcc419;">
            <b>{speedup:.1f}x FASTER</b>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html_overlay))

    # Save to project root to ensure README link works
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_file = os.path.join(project_root, "la_heatmap_demo.html")
    m.save(output_file)
    print(f"Done! Open {output_file} in your browser to see the demo.")

if __name__ == "__main__":
    if "H3_TURBO_LICENSE" in os.environ:
        h3_turbo.set_license_key(os.environ["H3_TURBO_LICENSE"].strip())
    run_demo()