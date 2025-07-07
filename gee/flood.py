import ee

ee.Authenticate()
ee.Initialize()

def get_flood_mask(region_geometry):
    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.products(dem).select("slope")
    low_elevation = dem.lt(200)
    flat_slope = slope.lt(10)
    flood_mask = low_elevation.And(flat_slope).selfMask()
    return flood_mask.clip(region_geometry)

def get_ndvi_mask(region_geometry):
    image = ee.ImageCollection("COPERNICUS/S2") \
        .filterBounds(region_geometry) \
        .filterDate("2023-01-01", "2023-12-31") \
        .sort("CLOUDY_PIXEL_PERCENTAGE") \
        .first()
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndvi_mask = ndvi.lt(0.2).selfMask()
    return ndvi_mask.clip(region_geometry)

def get_s1_water_mask(region_geometry):
    image = ee.ImageCollection("COPERNICUS/S1_GRD") \
        .filterBounds(region_geometry) \
        .filterDate("2023-01-01", "2023-12-31") \
        .filter(ee.Filter.eq("instrumentMode", "IW")) \
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")) \
        .select("VV") \
        .mean()
    water_mask = image.lt(-15).selfMask()
    return water_mask.clip(region_geometry)

# 2. Solar Irradiance (NASA POWER)
def get_solar_irradiance(region_geometry):
    solar = ee.ImageCollection("MODIS/061/MCD18A1") \
        .select("ALLSKY_SFC_SW_DWN") \
        .filterDate("2023-01-01", "2023-12-31") \
        .mean()
    return solar.clip(region_geometry)

# 3. Land Cover (ESA WorldCover 10m)
def get_land_cover(region_geometry):
    landcover = ee.Image("ESA/WorldCover/v100/2020")
    return landcover.clip(region_geometry)

# 4. Crop Growth Monitoring (Peak NDVI)
def get_peak_ndvi(region_geometry):
    collection = ee.ImageCollection("COPERNICUS/S2") \
        .filterDate("2023-01-01", "2023-12-31") \
        .filterBounds(region_geometry) \
        .map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI"))
    peak_ndvi = collection.max()
    return peak_ndvi.clip(region_geometry)
