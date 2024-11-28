from src.importation import os, json, numpy as np, tensorflow as tf, Dict, Tuple
from src.simulations.generators import generators_table
from src.distributions import distributions_table
from src.model import construct_model

optimizers_dict = {
    "adam": tf.optimizers.Adam
}

losses_dict = {
    "mse": tf.keras.losses.mean_squared_error,
    "mae": tf.keras.losses.mean_absolute_error,
    "mre": tf.keras.losses.mean_absolute_percentage_error
}

metrics_dict = {
    "mse": tf.keras.metrics.mean_squared_error,
    "mae": tf.keras.metrics.mean_absolute_error,
    "mre": tf.keras.metrics.mean_absolute_percentage_error
}

class Test_Evaluate(tf.keras.callbacks.Callback):
    """
    special callback to compute metrics on a given dataset
    (was very intense to implement for how essential it is)
    """
    def __init__(self, test_data, storage_path):
        self.test_data = test_data
        self.writer = tf.summary.create_file_writer(os.path.join(storage_path, "test"))

    def on_epoch_end(self, epoch, logs={}):
        super(Test_Evaluate, self).on_epoch_end(epoch, logs)

        scores = self.model.evaluate(self.test_data, verbose = False, return_dict = True)

        with self.writer.as_default():
            for key, value in scores.items():
                tf.summary.scalar(key, value, step=epoch)

        #print(" - ".join([f"test_{key}: {value}" for key, value in scores.items()]))
        
callbacks_dict = {
    "reduce_on_plateau": tf.keras.callbacks.ReduceLROnPlateau,
    "test_evaluate": Test_Evaluate
}

def load_dataset(dataset_name: str) -> Dict:
    """
    read a dataset description file

    -dataset_name: (string) name of the dataset description file

    return: (dictionary) the dataset
    """
    with open(os.path.join(
        "data", 
        "datasets", 
        f"{dataset_name}.json"
    )) as file:
        dataset = json.load(file)

    dataset["distributions"] = {
        variable: distributions_table[distribution_dict["type"]](**distribution_dict["parameters"])
        for variable, distribution_dict in dataset["distributions"].items()
    }

    return dataset

def load_experiment(
    experiment_name: str, 
    storage_path: str
) -> Tuple:
    """
    read an experiment description file

    -experiment_name: (string) name of the experiment description file

    return: (dictionary) the experiment
    """
    with open(os.path.join(
        "experiences", 
        "defined_experiences", 
        experiment_name + ".json"
    )) as file:
        experiment = json.load(file)
    
    experiment["storage_path"] = storage_path
    experiment["optimizer"] = optimizers_dict[experiment["optimizer"]](experiment["learning_rate"]) # set up the optimizer
    experiment["dataset"] = load_dataset(experiment["dataset"])
    experiment["model"] = construct_model(experiment["model"])

    for experiment_variable, experiment_variable_dict in zip(("losses", "metrics"), (losses_dict, metrics_dict)):
        experiment[experiment_variable] = {
            output_name: experiment_variable_dict[function] if experiment_variable == "losses" else [experiment_variable_dict[f] for f in function]
            for output_name, function in experiment[experiment_variable].items()
        }

    """
    Callbacks
    -first: record for each epoch
    -second: record for best epoch
    -third: record in the tensorboard
    -fourth: evaluate on dataset
    -others: defined in the experiment if any
    """
    experiment["callbacks"] = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(storage_path, "epoch_{epoch:02d}.h5"), 
            save_best_only = False, 
            save_freq = 5
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(storage_path, "best.h5"), 
            save_best_only = True
        ),
        tf.keras.callbacks.TensorBoard(storage_path),
        lambda x: Test_Evaluate(x, storage_path)
    ] + [
        callbacks_dict[callback](**parameters) 
        for callback, parameters in experiment["additional_callbacks"]
    ]
    del experiment["additional_callbacks"]

    # extract the inputs and outputs names and shapes (typical tensorflow nightmare stuff, forced harmless but ugly change (see last_name of denses))
    for product_type in ["inputs", "outputs"]:
        experiment[product_type] = {
            product_name: tf.TensorSpec(
                shape = [
                    dim for dim in (
                        getattr(
                            experiment["model"].get_layer(product_name), 
                            "input" if product_type == "inputs" else "output"
                        ).type_spec.shape
                    ) if (dim != None)
                ]
            )
            for product_name in experiment[product_type]
        }

    return experiment

def train(
    experiment_name: str,
    storage_path: str,
    seed: int,
    pretrain: str = None
) -> None:
    """
    Perform training from a experiment description file

    -experiment_name: (string) the name of the experiment description file
    -storage_path: (path) where training logs and model instances will be saved
    -seed: (integer) number for generating random variables
    """
    
    # see load_experiment.py
    experiment = load_experiment(
        experiment_name, 
        storage_path
    )

    dataset = experiment["dataset"]
    model = experiment["model"]
    inputs = experiment["inputs"]
    outputs = experiment["outputs"]

    rng_dataset = np.random.default_rng(seed) # generate seed for random variables
    tf.keras.utils.set_random_seed(seed) # fix tensorflow randomness (namely: weights initialization of the model)

    """
    depending on the dataset type, the training will proceed diferently:

    -generated: dataset is generate with functions and sets of pseudo-random parameters
    -spatial: dataset is extracted at pseudo-random locations in geolocated datas (vectorial and raster)
    -standard: dataset is loaded from an existing dataset
    """
    if dataset["type"] == "generated":
        # tf_datasets is a list of the train, validation and test dataset, may change in future development
        tf_datasets = generators_table[dataset["generation_type"]](inputs, outputs).cross_val(
            dataset,
            experiment["split"],
            experiment["sizes"], 
            rng_dataset
        )
    elif dataset["type"] == "spatial":
        # not implemented yet
        pass
    elif dataset["type"] == "standard":
        # not implemented yet
        pass

    training_dataset, validation_dataset, test_dataset = [
        tf_dataset.batch(experiment["batch_size"]).prefetch(tf.data.AUTOTUNE) 
        for tf_dataset in tf_datasets
    ]

    experiment["callbacks"][3] = experiment["callbacks"][3](test_dataset) # assign the test dataset to the test evaluate callback

    model.compile(
        optimizer = experiment["optimizer"],
        loss = experiment["losses"],
        metrics = experiment["metrics"]
    )

    if pretrain:
        model_name = f"group_{seed}"
        model_path = os.path.join("experiences", "trained_experiences", "train_generic", pretrain, model_name)
        model.load_weights(os.path.join(model_path, "best.h5"))

    model.fit(
        x = training_dataset,
        validation_data = validation_dataset,
        callbacks = experiment["callbacks"],
        epochs = experiment["epochs"],
        batch_size = experiment["batch_size"],
        verbose = 2
    )