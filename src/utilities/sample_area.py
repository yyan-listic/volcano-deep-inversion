from ..importation import rasterio, numpy as np, geopandas as gpd, shapely, json, os
from rasterio import windows
from typing import Tuple, List

def extract_samples(
    points: gpd.GeoSeries, 
    raster_filepath: str, 
    patchsize: int,
) -> list:
    extract_function = spatial_extract if patchsize else point_extract
    samples = extract_function(raster_filepath, points, patchsize)
    return samples

def spatial_extract(
    raster_filepath: str, 
    points: gpd.GeoSeries,
    patchsize: int
) -> List[np.ndarray]:
    
    with rasterio.open(raster_filepath) as raster:
        points_buffer = points.buffer(patchsize * raster.res[0])
        arrays = [raster.read(1, window=windows.from_bounds(point.bounds[0], point.bounds[1], point.bounds[2], point.bounds[3], raster.transform)) for point in points_buffer]
    return arrays

def point_extract(
    raster_filepath: str,
    points: gpd.GeoSeries
) -> np.ndarray:
    with rasterio.open(raster_filepath) as raster:
        values = raster.sample([(x,y) for x, y in zip(points.x, points.y)])
    return values

def line_points(
    line: shapely.LineString,
    step: float
) -> gpd.GeoSeries:
    return

def polygon_points(
    polygon: shapely.Polygon,
    step: float,
    offset: Tuple[float] = (0.,0.)
) -> gpd.GeoSeries:
    
    shape_envelope = polygon.bounds

    xs, ys = np.mgrid[shape_envelope[0]:shape_envelope[2]:step, shape_envelope[1]:shape_envelope[3]:step]
    xs = xs.flatten() + max(0, min(offset[0], step))
    ys = ys.flatten() + max(0, min(offset[1], step))
    
    points = gpd.GeoSeries([shapely.Point(x,y) for x, y in zip(xs, ys)])

    return points

def sample_area(
    area: str, 
    rasters: dict[str: str], 
    patchsize: int,
    step=0., 
    offset=(0.,0.)
) -> dict:
    with open(os.path.join("data", "areas", f"{area}.json"), "r") as area_file:
        area_dict = json.load(area_file)
    shape = shapely.wkt.loads(area_dict["shape"])

    if shape.geom_type == "MultiPoint":
        points = ""
    elif shape.geom_type == "MultiLineString":
        points = line_points(shape, step)
    elif shape.geom_type == "MultiPolygon":
        points = polygon_points(shape, step, offset)

    samples = {raster_name: extract_samples(points, raster_filepath, patchsize) for raster_name, raster_filepath in rasters.items()}
    
    return samples