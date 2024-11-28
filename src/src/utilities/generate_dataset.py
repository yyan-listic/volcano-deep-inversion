from ..importation import geopandas as gpd, shapely, os, json

def generate_dataset(
    name: str,
    select_shapes: gpd.GeoDataFrame,
    rasters: dict[str: str]
) -> None:
    """
    use shape to choose which areas from the "areas" directory to add in this dataset
    inputs are their name and not the raster file (is written in each area file)
    """
    shapes = []
    names = []
    for area_name in os.listdir(os.path.join("data","areas")):
        with open(os.path.join("data", "areas", f"{area_name}")) as area_file:
            area_dict = json.load(area_file)
        
        # need every raster to be specified in the area file
        if all([raster in area_dict["rasters"] for raster in rasters.keys()]):
            shapes.append(shapely.wkt.loads(area_dict["shape"]))
            names.append(area_name.rstrip(".json"))

    all_areas = gpd.GeoDataFrame({"name": names, "geometry": gpd.GeoSeries(shapes, crs=2056)})
    
    areas = list(gpd.sjoin(all_areas, select_shapes)["name"])

    dataset_dict = {
        "areas": areas,
        "rasters": rasters
    }
    dataset_json = json.dumps(dataset_dict)
    
    with open(os.path.join("data", "datasets", f"{name}.json"), "w") as dataset_file:
        dataset_file.write(dataset_json)

if __name__ == "__main__":
    
    from generate_area import shapefile_to_areas
    
    crs = 2056
    glacier_df = gpd.read_file("data/shapefiles/SGI_2016_glaciers.shp").to_crs(crs)
    rasters = {
        "slope": "data/rasters/slope_swissAlti3d.tif", 
        "ice_thickness": "data/rasters/IceThickness.tif", 
        "ice_velocity_magnitude": "data/rasters/V_RGI-11_2021July01.tif"
    }
    
    shapefile_to_areas(
        glacier_df,
        "sgi-id",
        rasters
    )

    for i in range(1,4):
        dataset_df = gpd.read_file(f"data/shapefiles/dataset_{i}.shp").to_crs(crs)

        generate_dataset(f"dataset_{i}", dataset_df, rasters)
