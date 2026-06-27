import ee
import geemap
import os
import json

ee.Initialize(project='pherl-ucdi')

def get_base_dir():
    return os.path.dirname(__file__)

roi = ee.Geometry.Rectangle([72.52, 23.01, 72.57, 23.05])

def write_metadata(dataset_name, band_names, scale, output_dir):
    metadata = {
        "dataset": dataset_name,
        "projection": "EPSG:4326",
        "resolution": f"{scale}m",
        "bands": band_names
    }
    meta_path = os.path.join(output_dir, f"{dataset_name}_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)

print("Task: Extracting GHSL...")
try:
    pop = ee.ImageCollection("JRC/GHSL/P2023A/GHS_POP").first().select(0).clip(roi).rename('Population_Density')
    built = ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_V").first().select(0).clip(roi).rename('Built_Up_Volume')
    combined_ghsl = pop.addBands(built)
    geemap.ee_export_image(combined_ghsl, filename=os.path.join('data', 'raw', 'ghsl', 'ghsl_socio.tif'), scale=50, region=roi)
except Exception as e:
    print(f"GHSL Extraction Failed: {e}")

print("Task: Extracting WorldPop...")
try:
    worldpop = ee.ImageCollection("WorldPop/GP/100m/pop").filterBounds(roi).first().clip(roi)
    geemap.ee_export_image(worldpop, filename=os.path.join('data', 'raw', 'worldpop', 'worldpop.tif'), scale=50, region=roi)
except Exception as e:
    print(f"WorldPop Extraction Failed: {e}")

print("Task: Extracting VIIRS Night Lights...")
try:
    viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate('2023-01-01', '2023-12-31').median().select('avg_rad').clip(roi)
    geemap.ee_export_image(viirs, filename=os.path.join('data', 'raw', 'viirs', 'viirs.tif'), scale=50, region=roi)
except Exception as e:
    print(f"VIIRS Extraction Failed: {e}")

print("GEE Supplementary Extraction Completed.")
