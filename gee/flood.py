import ee 

ee.Authenticate()  # Authenticate with Earth Engine
ee.Initialize()    # Initialize Earth Engine

# --- Flood-prone area mask using SRTM DEM and slope ---
def get_flood_mask(region_geometry):
    """
    Returns a mask for flood-prone areas based on low elevation and flat slope.
    Args:
        region_geometry: ee.Geometry object for the region of interest.
    Returns:
        ee.Image mask clipped to the region.
    """
    dem = ee.Image("USGS/SRTMGL1_003")  # SRTM DEM
    slope = ee.Terrain.products(dem).select("slope")  # Terrain slope
    low_elevation = dem.lt(200)  # Areas below 200m
    flat_slope = slope.lt(10)    # Areas with slope < 10 degrees
    flood_mask = low_elevation.And(flat_slope).selfMask()
    return flood_mask.clip(region_geometry)

# --- NDVI-based sparse vegetation mask ---
def get_ndvi_mask(region_geometry):

    # Returns a mask for areas with low NDVI (sparse vegetation).

    image = ee.ImageCollection("COPERNICUS/S2") \
        .filterBounds(region_geometry) \
        .filterDate("2023-01-01", "2023-12-31") \
        .sort("CLOUDY_PIXEL_PERCENTAGE") \
        .first()
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndvi_mask = ndvi.lt(0.2).selfMask()
    return ndvi_mask.clip(region_geometry)

# --- Water/flood extent using Sentinel-1 radar ---
def get_s1_water_mask(region_geometry):

    # Returns a mask for water/flood extent using Sentinel-1 VV polarization.

    image = ee.ImageCollection("COPERNICUS/S1_GRD") \
        .filterBounds(region_geometry) \
        .filterDate("2023-01-01", "2023-12-31") \
        .filter(ee.Filter.eq("instrumentMode", "IW")) \
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")) \
        .select("VV") \
        .mean()
    water_mask = image.lt(-15).selfMask()
    return water_mask.clip(region_geometry)

# --- Solar Irradiance (NASA POWER/MODIS) ---
def get_solar_irradiance(region_geometry):

    # Returns mean solar irradiance for the region.

    solar = ee.ImageCollection("MODIS/061/MCD18A1") \
        .select("ALLSKY_SFC_SW_DWN") \
        .filterDate("2023-01-01", "2023-12-31") \
        .mean()
    return solar.clip(region_geometry)

# --- Land Cover (ESA WorldCover 10m) ---
def get_land_cover(region_geometry):

    # Returns land cover classification for the region.

    landcover = ee.Image("ESA/WorldCover/v100/2020")
    return landcover.clip(region_geometry)

# --- Crop Growth Monitoring (Peak NDVI) ---
def get_peak_ndvi(region_geometry):

    # Returns the maximum NDVI value for the region in the year.
    
    collection = ee.ImageCollection("COPERNICUS/S2") \
        .filterDate("2023-01-01", "2023-12-31") \
        .filterBounds(region_geometry) \
        .map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI"))
    peak_ndvi = collection.max()
    return peak_ndvi.clip(region_geometry)
