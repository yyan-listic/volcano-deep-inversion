from src.importation import json, os
from src.load_model import load_model
from src.utilities.tf_load_dataset import load_dataset

def infer(experience, seed, save, dataset):

    experience_directory = os.path.join("experiences", experience)

    with open(os.path.join(experience_directory, f"{experience}.json")) as experience_file:
        experience_dict = json.load(experience_file)

    with open(os.path.join("models", f"{experience_dict['model']}.json")) as model_file:
        model_dict = json.load(model_file)

    # TODO!!!!
    rasters = model_dict["inputs"] + model_dict["outputs"]
    #patchsize = model_dict["patchsize"]
    #step = 0
    #offset = (0,0)

    inputs = []
    dataset = load_dataset(dataset)

    model = load_model(experience_dict['model'])
    model.load_weights(
        os.path.join(
            experience_directory,
            seed,
            "saves", 
            f"{save}.h5"
        )
    )

    # return or save somewhere ?
    outputs = model(dataset)