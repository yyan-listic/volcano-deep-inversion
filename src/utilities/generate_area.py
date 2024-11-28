from ..importation import geopandas as gpd, json, os

def generate_area(
    name: str, 
    wkt, 
    rasters: dict[str: str]
) -> None:

    area_json = json.dumps({
        "shape": wkt,
        "rasters": rasters
    })
    with open(os.path.join("data", "areas", f"{name}.json"), "w") as area_file:
        area_file.write(area_json)

def shapefile_to_areas(
    shapefile: gpd.GeoDataFrame,
    name_field: str,
    rasters: dict[str: str]
) -> None:
    """
    in this case, shapefile has already been open
    """

    for name, shape in zip(shapefile[name_field], shapefile["geometry"].to_wkt()):
        generate_area(name, shape, rasters)