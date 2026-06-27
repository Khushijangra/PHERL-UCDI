import osmnx as ox
import geopandas as gpd
import pandas as pd
import json
import os
from shapely.geometry import box

def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def write_metadata(dataset_name, features, output_dir):
    metadata = {
        "dataset": dataset_name,
        "source": "OpenStreetMap via OSMnx",
        "projection": "EPSG:4326 (Exported as GeoJSON)",
        "features": features
    }
    meta_path = os.path.join(get_base_dir(), 'data', 'metadata', f"{dataset_name}_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def extract_osm_data():
    print("Task 5: Extracting OpenStreetMap Data (Buildings & Roads)...")
    
    # Ahmedabad 5x5 km bounding box (same as GEE)
    # minx, miny, maxx, maxy (West, South, East, North)
    polygon = box(72.52, 23.01, 72.57, 23.05)
    
    out_dir = os.path.join(get_base_dir(), 'data', 'raw', 'osm')
    os.makedirs(out_dir, exist_ok=True)
    
    print("Fetching building footprints...")
    tags = {'building': True}
    try:
        buildings = ox.features_from_polygon(polygon, tags=tags)
        # Filter to only polygons/multipolygons
        buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        # Select essential columns to keep file size manageable
        cols_to_keep = ['geometry']
        if 'building:levels' in buildings.columns:
            cols_to_keep.append('building:levels')
        buildings = buildings[cols_to_keep]
        
        buildings_path = os.path.join(out_dir, 'buildings.geojson')
        buildings.to_file(buildings_path, driver='GeoJSON')
        print(f"Saved {len(buildings)} buildings to {buildings_path}")
    except Exception as e:
        print(f"Error fetching buildings: {e}")

    print("Fetching road network...")
    try:
        # Get drivable road network
        G = ox.graph_from_polygon(polygon, network_type='drive')
        
        # Convert graph to GeoDataFrames
        nodes, edges = ox.graph_to_gdfs(G)
        
        edges_path = os.path.join(out_dir, 'roads.geojson')
        edges[['geometry', 'highway', 'length']].to_file(edges_path, driver='GeoJSON')
        
        nodes_path = os.path.join(out_dir, 'intersections.geojson')
        nodes[['geometry']].to_file(nodes_path, driver='GeoJSON')
        
        print(f"Saved {len(edges)} roads and {len(nodes)} intersections.")
    except Exception as e:
        print(f"Error fetching roads: {e}")

    write_metadata("OSM_Morphology", ["Building Footprints", "Road Network", "Intersections"], out_dir)
    print("OSM Extraction Complete.")

if __name__ == "__main__":
    extract_osm_data()
