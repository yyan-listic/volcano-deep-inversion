from src.importation import numpy as np, os
from src.simulations.generators import generators_table
from src.experiment import load_experiment
from src.distributions import distributions_table

def infer_matrix(
    training_run,
    experiment_name,
    i,
    resolution,
    storage_path,
):
    experiment = load_experiment(experiment_name, storage_path)

    dataset = experiment["dataset"]
    model = experiment["model"]

    model_name = f"group_{i}"
    model_path = os.path.join("experiences", "trained_experiences", "train_generic", training_run, model_name)
    model.load_weights(os.path.join(model_path, "best.h5"))

    inputs = experiment["inputs"]
    outputs = experiment["outputs"]

    distributions = {
        "depth": distributions_table["uniform"](0, 4),
        "delta_volume": distributions_table["uniform"](0, 4),
    }

    if False:
        distributions = dataset["distributions"]

    tf_dataset = generators_table[dataset["generation_type"]](inputs, outputs).grid_iterate(
        dataset["fixed"],
        dataset["varyings"],
        distributions, 
        resolution
    ).batch(32)

    matrixes = {
        matrix_name: {output: [] for output in outputs} 
        for matrix_name in ["matrix", "dif_matrix", "dif_n_matrix"]
    }
    
    for tf_sample, ground_truth in tf_dataset:
        model_output = model(tf_sample)
        for output_name, output in model_output.items():
            matrixes["matrix"][output_name] += np.array(output).flatten().tolist()
            matrixes["dif_matrix"][output_name] += (np.array(output).flatten() - np.array(ground_truth[output_name]).flatten()).tolist()
            matrixes["dif_n_matrix"][output_name] += ((np.array(output).flatten() - np.array(ground_truth[output_name]).flatten()) / np.array(ground_truth[output_name]).flatten()).tolist()

    for matrix_name in ["matrix", "dif_matrix", "dif_n_matrix"]:
        for output_name, output in matrixes[matrix_name].items():
            np.save(
                os.path.join(storage_path, output_name + "_" + matrix_name), 
                np.array(output).reshape(*[resolution] * len(dataset["varyings"]))
            )
