import ee

ee.Authenticate()
ee.Initialize()

def get_flood_mask(region_geometry):
    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.products(dem).select("slope")
    low_elevation = dem.lt(300)
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

def get_population_overlay(region_geometry):
    pop = ee.ImageCollection("WorldPop/GP/100m/pop") \
        .filterDate("2020-01-01", "2020-12-31") \
        .first()
    return pop.clip(region_geometry)