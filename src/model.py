from src.importation import os, json, tensorflow as tf, List
from src.modules import modules_table

"""
everything to construct a model from a description file
"""

class Model_Branch:
    """
    a support for organizing operations in model construction
    """
    def __init__(
        self, 
        x_name: str, 
        y_name: str, 
        operations: List
    ) -> None:
        """
        -x_name: (string) the name of the input variable
        -y_name: (string) the name of the output variable
        -operations: (list) a serie of keras operations
        """
        self.x_name = x_name
        self.y_name = y_name
        self.operations = operations
    
    def __call__(self, x):
        """
        use the functional API of keras to stack operations
        """
        for operation in self.operations:
            x = operation(x)
        return x

def construct_model(
    model_name: str
) -> tf.keras.Model:
    """
    construct a model using a model description file

    -model_name: (string) name of the model description file

    return: (keras.Model) the model
    """
    with open(os.path.join("data", "models", f"{model_name}.json")) as model_file:
        model_dict = json.load(model_file)

    # set operations ready    
    operations_table = {
        operation_name: modules_table[operation["module"]](
            **operation["parameters"] if "parameters" in operation else {},  # if parameters then set the operation with it
            name = operation_name
        )
        for operation_name, operation in model_dict["operations"].items()
    }

    branches = [
        Model_Branch(
            branch["x_name"], 
            branch["y_name"], 
            [
                operations_table[operation_name] 
                for operation_name in branch["operations"]
            ]
        ) 
        for branch in model_dict["branches"]
    ]
    
    # will store every variables between each model branch (initialized with the inputs)
    xs = {
        key: tf.keras.layers.Input(input["size"], name = key) 
        for key, input in model_dict["inputs"].items()
    }

    for branch in branches:
        xs[branch.y_name] = branch(
            xs[branch.x_name] 
            if type(branch.x_name) == str else 
            [xs[x_name] for x_name in branch.x_name]
        )

    return tf.keras.Model(
        **{
            key: {x_name: xs[x_name] for x_name in model_dict[key]}
            for key in ["inputs", "outputs"]
        }
    )