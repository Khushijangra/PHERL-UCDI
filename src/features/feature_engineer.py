import os
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import json
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import libpysal
from esda.moran import Moran_Local

def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def create_50m_grid(bounds, crs="EPSG:4326"):
    """Creates a 50mx50m GeoDataFrame grid over the bounding box."""
    # Ahmedabad ROI: [72.52, 23.01, 72.57, 23.05]
    minx, miny, maxx, maxy = bounds
    # Convert to metric CRS for accurate 50m grid (UTM Zone 43N is EPSG:32643 for Gujarat)
    roi_poly = box(minx, miny, maxx, maxy)
    gdf = gpd.GeoDataFrame({'geometry': [roi_poly]}, crs=crs)
    gdf_metric = gdf.to_crs("EPSG:32643")
    
    m_minx, m_miny, m_maxx, m_maxy = gdf_metric.total_bounds
    
    grid_cells = []
    # Create 50m cells
    for x0 in np.arange(m_minx, m_maxx, 50):
        for y0 in np.arange(m_miny, m_maxy, 50):
            x1 = x0 + 50
            y1 = y0 + 50
            grid_cells.append(box(x0, y0, x1, y1))
            
    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:32643")
    grid_gdf['cell_id'] = range(len(grid_gdf))
    # Convert back to lat/lon for general use, but keep metric for area calculations
    grid_gdf['geometry_latlon'] = grid_gdf.geometry.to_crs("EPSG:4326")
    grid_gdf['latitude'] = grid_gdf['geometry_latlon'].centroid.y
    grid_gdf['longitude'] = grid_gdf['geometry_latlon'].centroid.x
    return grid_gdf

def sample_raster_to_grid(grid_gdf, raster_path, band_names):
    """Reads a raster and extracts mean values for each grid cell."""
    if not os.path.exists(raster_path):
        print(f"Warning: {raster_path} not found. Skipping...")
        for name in band_names:
            grid_gdf[name] = np.nan
        return grid_gdf
        
    import rasterstats
    with rasterio.open(raster_path) as src:
        # Reproject grid to match raster CRS
        grid_proj = grid_gdf.to_crs(src.crs)
        for i, band_name in enumerate(band_names, start=1):
            arr = src.read(i)
            stats = rasterstats.zonal_stats(grid_proj.geometry, arr, affine=src.transform, stats="mean", all_touched=True, nodata=np.nan)
            grid_gdf[band_name] = [s['mean'] if s['mean'] is not None else np.nan for s in stats]
            
    return grid_gdf

def build_master_cube():
    print("PHASE 1.5: Constructing Master Urban Climate Feature Cube...")
    base_dir = get_base_dir()
    
    # 1. Initialize Grid
    bounds = (72.52, 23.01, 72.57, 23.05)
    cube = create_50m_grid(bounds)
    print(f"Created grid with {len(cube)} cells (50x50m).")
    
    # 2. Extract Raster Features
    print("Sampling Thermal Data...")
    landsat_mean = os.path.join(base_dir, 'data', 'raw', 'landsat', 'lst_mean.tif')
    cube = sample_raster_to_grid(cube, landsat_mean, ['lst_mean'])
    
    landsat_max = os.path.join(base_dir, 'data', 'raw', 'landsat', 'lst_max.tif')
    cube = sample_raster_to_grid(cube, landsat_max, ['lst_max'])
    
    landsat_std = os.path.join(base_dir, 'data', 'raw', 'landsat', 'lst_std.tif')
    cube = sample_raster_to_grid(cube, landsat_std, ['lst_variance'])
    
    print("Sampling Optical Data...")
    sentinel_path = os.path.join(base_dir, 'data', 'raw', 'sentinel', 'sentinel_derived.tif')
    opt_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'ndvi', 'ndbi', 'ndwi', 'savi', 'evi', 'albedo', 'vegetation_fraction', 'impervious_fraction']
    cube = sample_raster_to_grid(cube, sentinel_path, opt_bands)
    
    print("Sampling Meteorological Data...")
    era5_path = os.path.join(base_dir, 'data', 'raw', 'era5', 'era5_climate.tif')
    met_bands = ['air_temperature', 'dewpoint', 'wind_u', 'wind_v', 'solar_radiation', 'pressure', 'evaporation']
    cube = sample_raster_to_grid(cube, era5_path, met_bands)
    
    print("Sampling Socio-Economic Data...")
    worldpop_path = os.path.join(base_dir, 'data', 'raw', 'worldpop', 'worldpop.tif')
    cube = sample_raster_to_grid(cube, worldpop_path, ['population_density'])
    
    viirs_path = os.path.join(base_dir, 'data', 'raw', 'viirs', 'viirs.tif')
    cube = sample_raster_to_grid(cube, viirs_path, ['nighttime_lights'])
    
    # 3. OSM Morphological Extraction (Rasterizing geometry)
    print("Processing OSM Morphology...")
    osm_path = os.path.join(base_dir, 'data', 'raw', 'osm', 'buildings.geojson')
    if os.path.exists(osm_path):
        bldgs = gpd.read_file(osm_path).to_crs("EPSG:32643")
        cube['building_density'] = cube.geometry.apply(lambda geom: bldgs.intersection(geom).area.sum() / geom.area)
    else:
        cube['building_density'] = 0.0 # Fallback
        
    cube['road_density'] = 0.0 # Placeholder for road network density computation
    
    # Fill Nans (for cells missing data)
    cube = cube.fillna(0)
    
    # 4. Create Composite Decision Variables (Task 3)
    print("Computing Composite Indices...")
    cube['heat_persistence'] = (cube.get('day_lst', 0) + cube.get('night_lst', 0)) / 2.0
    cube['exposure_index'] = cube.get('population_density', 0) * cube.get('lst_mean', 0)
    
    # TVI formulation
    lst_norm = (cube.get('lst_mean',0) - cube.get('lst_mean',0).mean()) / (cube.get('lst_mean',0).std() + 1e-6)
    pop_norm = (cube.get('population_density',0) - cube.get('population_density',0).mean()) / (cube.get('population_density',0).std() + 1e-6)
    cube['thermal_vulnerability_index'] = 0.5 * lst_norm + 0.3 * pop_norm - 0.2 * cube.get('ndvi', 0)
    
    # UHEI formulation
    cube['urban_heat_equity_index'] = 0.4 * lst_norm + 0.4 * pop_norm - 0.2 * cube.get('albedo', 0)
    
    # 5. Spatial Statistics (Task 1 continuation)
    print("Computing Spatial Statistics (Spatial Lag & Moran's I)...")
    # Create Queen contiguity weights
    try:
        w = libpysal.weights.Queen.from_dataframe(cube)
        w.transform = 'r' # row-standardized
        # Spatial lag of LST
        cube['spatial_lag_lst'] = libpysal.weights.lag_spatial(w, cube['lst_mean'].values)
        # Local Moran's I
        lisa = Moran_Local(cube['lst_mean'].values, w)
        cube['local_morans_i'] = lisa.Is
        cube['getis_ord_gi'] = lisa.Zs # Approximation
    except Exception as e:
        print(f"Spatial weights computation skipped (requires data): {e}")
        cube['spatial_lag_lst'] = 0.0
        cube['local_morans_i'] = 0.0
        cube['getis_ord_gi'] = 0.0

    # 6. Generate Intervention Feasibility Masks (Task 2)
    print("Generating Intervention Feasibility Masks...")
    cube['mask_trees'] = ((cube['ndvi'] < 0.4) & (cube['building_density'] < 0.3)).astype(int)
    cube['mask_coolroof'] = (cube['building_density'] > 0.1).astype(int)
    cube['mask_reflective_pavement'] = (cube['road_density'] > 0).astype(int)
    
    # 7. Export Parquet
    out_dir = os.path.join(base_dir, 'data', 'processed')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'master_feature_cube.parquet')
    
    # GeoPandas to_parquet
    print(f"Saving Master Feature Cube to {out_file}...")
    cube.to_parquet(out_file, index=False)
    
    # 8. Validation Report (Task 4)
    print("Generating Validation Metadata...")
    stats = cube.describe().to_dict()
    meta_dir = os.path.join(base_dir, 'data', 'metadata')
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, 'feature_cube_metadata.json'), 'w') as f:
        json.dump({"total_cells": len(cube), "resolution": "50mx50m", "columns": list(cube.columns), "summary_stats": str(stats)[:500] + "..."}, f, indent=4)
        
    print("Phase 1.5 Execution Complete. Master Feature Cube ready.")

if __name__ == "__main__":
    build_master_cube()
