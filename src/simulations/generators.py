from src.importation import tensorflow as tf, numpy as np, Dict, List, Tuple
from src.split_intervals import random_sample, excluding_intervals
from src.simulations.volcanic_deformation import Observator, Mogi, Volcanic_Deformation

"""
elements for handling 
"""

class Tf_Generator:
    """
    a class that handle datasets for tensorflow Model without everying loaded in memory at once
    may cost much cpu usage until correct batch processing is implemented, sorry
    used for training as well as inference
    """
    def __init__(
        self, 
        inputs: Dict, 
        outputs: Dict
    ) -> None:
        """
        -inputs/outputs: (dictionary[string: tf.TensorSpec]) dictionaries of variables with their tensor types/shapes
        
        note: TensorSpec is a difficult variable to manipulate 
        try to get it from the model layers rather than setting it manually
        """
        self.inputs = inputs
        self.outputs = outputs

    def random_split(
        self, 
        varyings: Dict, 
        split: List, 
        rng: np.random.Generator
    ) -> List:
        """
        create excluding intervals for cross validation 
        (see excluding_intervals documentation)
        
        kind of a useless method but could be improved in the future
        """
        return excluding_intervals(varyings, split, 100., rng)

    def tf_sample(self, sample: Dict) -> Tuple:
        """
        transform a convenient dictionary with all datas into a tuple of inputs and outputs separated
        
        -sample: (dictionary[string: tf.Tensor]) dictionary with all datas generated

        return: (tuple) tuple with inputs dictionary on 0 and dictionary datas on 1
        """
        return tuple(
            [
                {
                    product_name: sample[product_name] 
                    for product_name in products.keys()
                }
                for products in (self.inputs, self.outputs)
            ]
        )

    def grid_iterate(
        self, 
        fixed: Dict, 
        varyings: List, 
        distributions: Dict, 
        resolution: int
    ) -> tf.data.Dataset:
        """
        get every combination of parameters of a dataset
        may fuse with Tf_Generator.random_iterate() in later versions
        
        -fixed: (dictionary[string: Any]) dataset variables that don't vary and their constant value
        -varyings: (list) dataset variables that do vary
        -distributions: (dictionary[string: distributions.Distribution]) every variable's distributions
        -resolution: number of elements on each axis (may change in future versions to allow different resolution for each axis)

        return: (tf.data.Dataset) samples iterator based on tf.data.Dataset.from_generator

        note: according to some users, tf.data.Dataset.from_generator induce extreme memory usage
        however i did not experienced such issue and memory usage stood low for every try
        """
        
        parameters = {**fixed}  # quick shallow duplicate of fixed

        def iterator():
            """
            quick ghost function that yield samples
            """
            for i in range(resolution ** len(varyings)):  # for every sample
                for j, key in enumerate(varyings):
                    quantile = int(i / resolution ** j) / resolution % 1  # quantile is incremented at different rate for each variable
                    parameters[key] = distributions[key].qf(quantile)  # variable's quantile to variable's value
                sample = self(**parameters)  # turn set of parameters into variable (__call__() method need to be defined in inheritance of this class)
                yield self.tf_sample(sample)
        
        return tf.data.Dataset.from_generator(
            iterator,
            output_signature = (
                {k: v for k, v in self.inputs.items()},
                {k: v for k, v in self.outputs.items()}
            )
        )

    def random_iterate(
        self, 
        fixed: Dict, 
        intervals: Dict, 
        distributions: Dict, 
        size: int, 
        rng: np.random.Generator
    ) -> tf.data.Dataset:
        """
        produce a dataset generator that randomly generate samples

        -fixed: (dictionary[string: Any]) dataset variables that don't vary and their constant value
        -intervals: (dictionary[string: list[list[float]]]) dataset variables that do vary and their quantile intervals
        -distributions: (dictionary[string: distributions.Distribution]) every variable's distributions
        -size: (integer) number of samples to draw from the dataset

        return: (tf.data.Dataset) samples iterator based on tf.data.Dataset.from_generator
        """
        
        parameters = {**fixed}  # quick shallow copy

        def iterator():
            """
            quick ghost function that yields elements
            """
            for _ in range(size):
                for key, quantile in random_sample(intervals, rng).items():
                    parameters[key] = distributions[key].qf(quantile)
                sample = self(**parameters)
                yield self.tf_sample(sample)

        return tf.data.Dataset.from_generator(
            iterator,
            output_signature = (
                {k: v for k, v in self.inputs.items()},
                {k: v for k, v in self.outputs.items()}
            )
        )

    def cross_val(
        self, 
        dataset: Dict, 
        split: List, 
        sizes: List, 
        rng: np.random.Generator
    ) -> Tuple:
        """
        generate several datasets generators
        the last dataset is not randomly generated
        more info in Generator.random_split()
        
        -dataset: (dictionary[string: Any]) the global dataset description
        -split: (list[float]) the fraction associated with every datasets except the last, sum of elements must be 1
        -sizes: (list[int]) the number of elements that each dataset generator will randomly produce, length must be same as split + 1

        return: (tuple[tf.data.Dataset]) 
        """
        return (
            self.random_iterate(
                dataset["fixed"], 
                intervals, 
                dataset["distributions"],
                size, 
                rng
            ) for size, intervals in zip(
                sizes, 
                self.random_split(
                    dataset["varyings"], 
                    split, 
                    rng
                )
            )
        )

class Tf_Volcanic_Generator(Tf_Generator):
    """
    an abstract class for every data simulation related to volcanic deformations
    will produce deformation images from sets of parameters

    one set is related to satellite trajectory (see the Observator) 
    the other for volcanic deformation (see Volcanic_Deformation and related class)
    """
    def __call__(
        self, 
        volcanic_deformation: Volcanic_Deformation, 
        azimuth: float, 
        incidence: float, 
        x_res: float, 
        y_res: float,
        azimuth_1: float = None,
        incidence_1: float = None,
        x_res_1: float = None,
        y_res_1: float = None
    ) -> Dict:
        observator = Observator(azimuth, incidence, x_res, y_res)
        grid_deformation = volcanic_deformation.grid(observator)  # generate the displacement image
        sample = {
            "displacement": np.expand_dims(grid_deformation, axis = -1),  # just add that 
            "azimuth": [azimuth],
            "incidence": [incidence],
            "x_res": [x_res],
            "y_res": [y_res]
        }
        if np.all([a is not None for a in [azimuth_1, incidence_1, x_res_1, y_res_1]]):
            second_observator = Observator(azimuth_1, incidence_1, x_res_1, y_res_1)
            second_grid_deformation = volcanic_deformation.grid(second_observator)  # generate the displacement image
            sample = {
                **sample,
                "displacement_1": np.expand_dims(second_grid_deformation, axis = -1),  # just add that 
                "azimuth_1": [azimuth_1],
                "incidence_1" : [incidence_1],
                "x_res_1" : [x_res_1],
                "y_res_1" : [y_res_1]
            }

        return sample

class Tf_Mogi_Generator(Tf_Volcanic_Generator):
    """
    class for deformation based on mogi model

    see Tf_Volcanic_Generator and Mogi class for more details
    """
    def __call__(
        self, 
        depth: float, 
        delta_volume: float, 
        x: float, 
        y: float, 
        noise = False, 
        **kwargs
    ) -> Dict:
        mogi = Mogi(depth, delta_volume, x, y, noise)
        sample = super(Tf_Mogi_Generator, self).__call__(mogi, **kwargs)
        sample["depth"] = [depth]
        sample["delta_volume"] = [delta_volume]
        sample["x"] = [x]
        sample["y"] = [y]
        sample["noise"] = [noise]
        return sample
    
# if you create new datasets generators, be sure to write them here to access them
generators_table = {
    "mogi": Tf_Mogi_Generator
}