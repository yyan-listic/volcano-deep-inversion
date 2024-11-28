from ..importation import tensorflow as tf, os, json
from .sample_area import sample_area
from typing import List

def load_dataset(
    dataset: str,
    inputs: List[str],
    outputs: List[str],
    patchsize: float,
    step: float
) -> dict:
    
    with open(os.path.join("data", "datasets", f"{dataset}.json")) as dataset_file:
        dataset_dict = json.load(dataset_file)
    
    areas = dataset_dict["areas"]
    rasters = dataset_dict["rasters"]
    offset = (0., 0.)

    rasters_kept = {raster_name: raster_path for raster_name, raster_path in rasters.items()}

    tf_sample_area = lambda area: tf.data.Dataset.from_tensor_slices(
        tf.py_function(
            lambda x: sample_area(
                str(x.numpy())[2:-1], 
                rasters_kept, 
                patchsize, 
                step, 
                offset
            ),
            [area], 
            [tf.float32]
        )
    )

    tf_dataset = tf.data.Dataset.from_tensor_slices(areas)
    tf_dataset = tf_dataset.interleave(tf_sample_area)

    @tf.function
    def decoder(sample: dict) -> dict:
        xs = {}
        ys = {}
        for name, value in sample:
            if name in inputs:
                xs[name] = value
            if name in outputs:
                ys[name] = value
        return (xs, ys)

    #tf_dataset = tf_dataset.map(decoder)

    return tf_dataset

if __name__ == "__main__":

    for x in load_dataset(
        "dataset_1",
        ["slope"],
        ["ice_thickness"],
        100,
        100
    ):
        print(x)