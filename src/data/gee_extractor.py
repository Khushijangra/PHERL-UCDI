import ee
import geemap
import os
import json

# Initialize Earth Engine.
try:
    ee.Initialize(project='pherl-ucdi')
    print("Earth Engine initialized successfully.")
except Exception as e:
    print(f"Earth Engine initialization failed. Error: {e}")
    exit()

def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def define_ahmedabad_roi():
    """
    Defines a 5x5 km Region of Interest (ROI) in West/Central Ahmedabad.
    """
    return ee.Geometry.Rectangle([72.52, 23.01, 72.57, 23.05])

def write_metadata(dataset_name, band_names, scale, output_dir):
    metadata = {
        "dataset": dataset_name,
        "acquisition_period": "2023-04-01 to 2023-06-30 (Summer)",
        "projection": "EPSG:4326",
        "resolution": f"{scale}m",
        "bands": band_names
    }
    meta_path = os.path.join(get_base_dir(), 'data', 'metadata', f"{dataset_name}_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def extract_landsat(roi, start_date='2023-04-01', end_date='2023-06-30'):
    """Task 1: Landsat Thermal Extraction"""
    print("Task 1: Extracting Landsat 8/9 Thermal Data...")
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filterDate(start_date, end_date)
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(roi).filterDate(start_date, end_date)
    dataset = l8.merge(l9).filter(ee.Filter.lt('CLOUD_COVER', 10))
    
    def apply_scale(image):
        lst = image.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')
        return lst

    lst_col = dataset.map(apply_scale)
    
    lst_mean = lst_col.mean().rename('LST_Mean').clip(roi)
    lst_max = lst_col.max().rename('LST_Max').clip(roi)
    lst_std = lst_col.reduce(ee.Reducer.stdDev()).rename('LST_Std').clip(roi)
    
    out_dir = os.path.join(get_base_dir(), 'data', 'raw', 'landsat')
    geemap.ee_export_image(lst_mean, filename=os.path.join(out_dir, 'lst_mean.tif'), scale=50, region=roi)
    geemap.ee_export_image(lst_max, filename=os.path.join(out_dir, 'lst_max.tif'), scale=50, region=roi)
    geemap.ee_export_image(lst_std, filename=os.path.join(out_dir, 'lst_std.tif'), scale=50, region=roi)
    write_metadata("Landsat_Thermal", ["LST_Mean", "LST_Max", "LST_Std"], 50, out_dir)

def extract_ecostress(roi, start_date='2023-04-01', end_date='2023-06-30'):
    """Task 2: ECOSTRESS Diurnal Extraction"""
    print("Task 2: Extracting ECOSTRESS Thermal Data...")
    # ECOSTRESS LST (ECO2LSTE.001) is available in GEE
    dataset = ee.ImageCollection("NASA/ECOSTRESS/V002/L2_LSTE") \
                .filterBounds(roi) \
                .filterDate(start_date, end_date)
    
    def apply_scale(img):
        # LST is scaled by 0.02
        return img.select('LST').multiply(0.02).subtract(273.15).copyProperties(img, ['system:time_start'])
        
    lst_col = dataset.map(apply_scale)
    
    # Very rudimentary day/night split based on hour of day (UTC -> IST +5:30)
    # GEE time is UTC. Daytime in IST (e.g. 8AM-5PM) is roughly 02:30-11:30 UTC.
    day_col = lst_col.filter(ee.Filter.calendarRange(3, 11, 'hour'))
    night_col = lst_col.filter(ee.Filter.calendarRange(13, 23, 'hour'))
    
    lst_day = day_col.mean().rename('LST_Day').clip(roi)
    lst_night = night_col.mean().rename('LST_Night').clip(roi)
    diurnal_diff = lst_day.subtract(lst_night).rename('LST_Diurnal_Diff')
    
    out_dir = os.path.join(get_base_dir(), 'data', 'raw', 'ecostress')
    combined = lst_day.addBands([lst_night, diurnal_diff])
    # GEE might throw errors if Day/Night collections are empty for the small date range. Handle gracefully.
    try:
        geemap.ee_export_image(combined, filename=os.path.join(out_dir, 'ecostress_diurnal.tif'), scale=50, region=roi)
        write_metadata("ECOSTRESS", ["LST_Day", "LST_Night", "LST_Diurnal_Diff"], 50, out_dir)
    except Exception as e:
        print(f"Warning: ECOSTRESS extraction failed (likely no overpasses in date range). Error: {e}")

def extract_sentinel(roi, start_date='2023-04-01', end_date='2023-06-30'):
    """Task 3: Sentinel-2 Optical Extraction"""
    print("Task 3: Extracting Sentinel-2 Optical Data...")
    dataset = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
                .filterBounds(roi) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    
    def process_bands(image):
        b2 = image.select('B2').multiply(0.0001)
        b3 = image.select('B3').multiply(0.0001)
        b4 = image.select('B4').multiply(0.0001)
        b8 = image.select('B8').multiply(0.0001)
        b11 = image.select('B11').multiply(0.0001)
        b12 = image.select('B12').multiply(0.0001)
        
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        savi = b8.subtract(b4).multiply(1.5).divide(b8.add(b4).add(0.5)).rename('SAVI')
        evi = b8.subtract(b4).multiply(2.5).divide(b8.add(b4.multiply(6.0)).subtract(b2.multiply(7.5)).add(1.0)).rename('EVI')
        albedo = b2.multiply(0.356).add(b4.multiply(0.130)).add(b8.multiply(0.373)).add(b11.multiply(0.085)).add(0.018).rename('Albedo')
        
        # Approximate fractions (Linear unmixing approximation)
        veg_frac = ndvi.subtract(0.1).divide(0.7).clamp(0, 1).rename('Vegetation_Fraction')
        imp_frac = ndbi.subtract(-0.2).divide(0.4).clamp(0, 1).rename('Impervious_Fraction')
        
        return b2.rename('B2').addBands([b3.rename('B3'), b4.rename('B4'), b8.rename('B8'), b11.rename('B11'), b12.rename('B12'), 
                                         ndvi, ndbi, ndwi, savi, evi, albedo, veg_frac, imp_frac])

    processed = dataset.map(process_bands).median().clip(roi)
    
    out_dir = os.path.join(get_base_dir(), 'data', 'raw', 'sentinel')
    geemap.ee_export_image(processed, filename=os.path.join(out_dir, 'sentinel_derived.tif'), scale=50, region=roi)
    bands = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDBI", "NDWI", "SAVI", "EVI", "Albedo", "Vegetation_Fraction", "Impervious_Fraction"]
    write_metadata("Sentinel_2", bands, 50, out_dir)

def extract_era5(roi, start_date='2023-04-01', end_date='2023-06-30'):
    """Task 4: ERA5 Meteorology Extraction"""
    print("Task 4: Extracting ERA5 Climate Data...")
    dataset = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterBounds(roi).filterDate(start_date, end_date)
    
    bands = ['temperature_2m', 'dewpoint_temperature_2m', 'u_component_of_wind_10m', 'v_component_of_wind_10m', 
             'surface_solar_radiation_downwards', 'surface_pressure', 'total_evaporation']
    mean_climate = dataset.select(bands).mean().clip(roi)
    
    out_dir = os.path.join(get_base_dir(), 'data', 'raw', 'era5')
    geemap.ee_export_image(mean_climate, filename=os.path.join(out_dir, 'era5_climate.tif'), scale=50, region=roi)
    write_metadata("ERA5_Land", bands, 50, out_dir)

def extract_ghsl(roi):
    """Task 6: GHSL Extraction"""
    print("Task 6: Extracting GHSL (Population & Settlement) Data...")
    # GHSL Population 2020
    pop = ee.Image("JRC/GHSL/P2030A/GHS_POP/2020").select(0).clip(roi).rename('Population_Density')
    # GHSL Built-up 2020
    built = ee.Image("JRC/GHSL/P2030A/GHS_BUILT_V/2020").select(0).clip(roi).rename('Built_Up_Volume')
    
    combined = pop.addBands(built)
    out_dir = os.path.join(get_base_dir(), 'data', 'raw', 'ghsl')
    geemap.ee_export_image(combined, filename=os.path.join(out_dir, 'ghsl_socio.tif'), scale=50, region=roi)
    write_metadata("GHSL", ["Population_Density", "Built_Up_Volume"], 50, out_dir)

if __name__ == "__main__":
    roi = define_ahmedabad_roi()
    
    extract_landsat(roi)
    extract_ecostress(roi)
    extract_sentinel(roi)
    extract_era5(roi)
    extract_ghsl(roi)
    
    print("GEE Extraction Script Completed.")
