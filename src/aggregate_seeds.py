# Aggregate some info from all the seeds of an experiment

import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import copy

if __name__ == "__main__":

    # Read all the subdirectories starting with group_
    list_seed_dirs = [
        file for file in os.listdir(args.storage_path)
        if file.startswith("group_")
    ]

    # Read experience name from info.yaml
    with open(os.path.join(args.storage_path, 'info.yaml'), 'rb') as f:
        info = yaml.safe_load(f)
    experience_name = info['groups_of_parameters'][0]['pos_0']

    # Read data from all
    log_data = {
        "epoch_dense_1_mae": [],
        "epoch_dense_1_mse": [],
        "epoch_dense_3": [],
        "epoch_dense_3": [],
        "epoch_loss": []
    }
    log_data_train = copy.deepcopy(log_data)
    log_data_validation = copy.deepcopy(log_data)
    for name, log_data in zip(("train", "validation"), (log_data_train, log_data_validation)):

        log_folder = os.path.join(model_path, folder, "logs", name)
        log_file = os.path.join(log_folder, os.listdir(log_folder)[0])

        parser = argparse.ArgumentParser(description="Aggregates result of an experiment over seed")
        parser.add_argument("--storage_path", type=str, required=True)
        args = parser.parse_args()
    
    event_file_iterator = tf.python.summary.summary_iterator(log_file)

    # Loop over all the events in the event file
    for event in event_file_iterator:
        # Check if the event is a summary event
        if event.summary:
            # Loop over all the values in the summary event
            for value in event.summary.value:
                t = tf.make_ndarray(value.tensor)
                if value.tag in log_data.keys():
                    log_data[value.tag].append(t)