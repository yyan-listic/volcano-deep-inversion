from typing import Any
from src.importation import tensorflow as tf, List

"""
a script that gather every keras layers that any model should use

if you want to add a custom one, follow these guidelines:

-complex layers: 
    -when to use: your layer is not supported by keras Sequential (e.g: multiple inputs, data splitted in parallel processings)
    -what to use: make a class inherited from Module

-sequence layers: 
    -when to use: your layer is a straight serie of operations
    -what to use: make a simple function that creates it (see template below)
    
    def sequence_layer(name: str = None):  # the "name" variable should always be kept
        operations = [operation1, operation2, ...]  # operations are keras.Layer
        return tf.Sequential(operations)

after you have written your layer or if you are adding an existing one from keras, add it to modules_table at the end of the script
"""

class Module:
    """
    this is an abstract class for any custom layer that is too complex to be written as a keras.Sequential
    i can't decide what should be the correct way to inherit it, so just follow the documentation in each method
    """
    def __init__(self, name: str = None) -> None:
        # the "name" variable should always be featured (you can ignore the variable and also add other variables)
        pass

    def __call__(self, x) -> Any:
        # don't add any other variable, "x" is the input of your layer and you seriously don't need anything else here (if you have two inputs, concatenate them)
        pass

class Resnet_Block(Module):
    """
    the traditional ResNet block composed of two elements:

    -a sequence of Conv2D with kernel of size 3
    -a "shortcut" (a Conv2D with kernel of size 1)

    the two elements are then added together term by term
    """
    def __init__(self, filters: List, name: str = None) -> None:
        self.convolutions = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters = filter,
                    kernel_size = 3,
                    strides = 1,
                    padding = "same",
                    activation = "leaky_relu" if i != len(filters) - 1 else None,  # in final conv activation does not occur before the addition with the shortcut
                    kernel_regularizer = "l2",
                    kernel_initializer = "glorot_uniform",
                    bias_initializer = "zeros",
                    name = f"{name}_convolutions_{i}" if name else None
                ) for i, filter in enumerate(filters)
            ],
            name = name
        )

        self.shortcut = tf.keras.layers.Conv2D(
            filters = filters[-1],
            kernel_size = 1,
            strides = 1,
            activation = None,
            kernel_regularizer = "l2",
            kernel_initializer = "glorot_uniform",
            bias_initializer = "zeros",
            name = f"{name}_shortcut" if name else None
        )

        self.concatenate = tf.keras.layers.Concatenate(
            name = f"{name}_concatenation" if name else None
        )
    
    def __call__(self, x):
        return self.concatenate([
            self.convolutions(x), 
            self.shortcut(x)
        ])

def denses(
    filters: List, 
    name: str = None
) -> tf.keras.Sequential:
    """
    create a group of dense layers to use in a model

    -filters: (list) the units sizes for each dense layer
    -last_activation: (string) the name of the last activation to be used
    -last_name: (string) the name of the last dense layer, see note below
    -name: (string) the name of the denses group (default to None, every layer will have default name)
    
    return: (keras.Sequential) a sequence of dense layers
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(
                name = f"{name}_flatten" if name else None
            )
        ] + [
            tf.keras.layers.Dense(
                units = filter,
                activation = "leaky_relu" if i != len(filters) - 1 else None,
                use_bias = False,
                kernel_initializer = "glorot_uniform",
                kernel_regularizer = "l2",
                name = f"{name}_dense_{i}" if name else None
            ) for i, filter in enumerate(filters)
        ],
        name = name
    )

class Attention(Module):
    def __init__(
        self, 
        filters: List,
        name: str = None
    ) -> None:
        """
        """
        self.attention_branch = tf.keras.Sequential(
            [
                tf.keras.layers.GlobalAveragePooling2D(
                    name = f"{name}_global_average"
                ),
                tf.keras.layers.Flatten(
                    name = f"{name}_flatten"
                )
            ] + [
                tf.keras.layers.Dense(
                    units = filter,
                    activation = "leaky_relu" if i == 0 else "sigmoid",
                    use_bias = True,
                    kernel_regularizer = "l2",
                    kernel_initializer = "glorot_uniform",
                    bias_initializer = "zeros",
                    name = f"{name}_dense_{i}" if name else None

                ) for i, filter in enumerate(filters)
            ]
        )

        self.multiply = tf.keras.layers.Multiply(
            name = f"{name}_multiply"
        )
    
    def __call__(self, x) -> Any:
        return self.multiply([
            x, 
            self.attention_branch(x)
        ])

# all types of operations available (don't forget to add your new ones here)
modules_table = {
    "max_pooling": tf.keras.layers.MaxPooling2D,
    "concatenate": tf.keras.layers.Concatenate,
    "activation": tf.keras.layers.Activation,
    "denses": denses,
    "resnet_block": Resnet_Block,
    "attention": Attention,
    "concatenation": tf.keras.layers.Concatenate
}
