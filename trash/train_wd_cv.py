from keras.src.applications import ResNet50
from scipy.stats import wasserstein_distance
import numpy as np
import re
import os
import pandas as pd
from src.data.data_prep import prepare_cifar10_complete, comparison_datasets_names, prepare_dataset, prepare_cifar10_c_complete


def extract_features(dataset):
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(32, 32, 3))

    features = base_model.predict(dataset, batch_size=32, verbose=1)

    return features


def compute_wasserstein_distance(features1, features2):
    wd_distances = []
    for i in range(features1.shape[1]):
        wd = wasserstein_distance(features1[:, i], features2[:, i])
        wd_distances.append(wd)

    return np.mean(wd_distances)


output_file = os.path.join(os.getcwd(), "log_likelihood_results.txt")
original_train_ds = prepare_cifar10_complete()
comparison_datasets = prepare_cifar10_c_complete(flatten=False, remove_labels=False)

df_arr = []
for run in range(10):
    original_features = extract_features(original_train_ds)
    comparison_features_arr = []
    for index, dataset in enumerate(comparison_datasets):
        comparison_features = extract_features(dataset)
        comparison_features_arr.append({"features": comparison_features, "corruption": comparison_datasets_names[index]})

    wd_distances = []
    for index, features in enumerate(comparison_features_arr):
        wd_distance = compute_wasserstein_distance(original_features, features["features"])
        wd_distances.append(wd_distance)

        pattern = re.compile(r'^(.*)_(\d+)$')
        match = pattern.match(features['corruption'])
        corruption = match.group(1)
        severity = match.group(2)

        df_arr.append({
            "run": run,
            "corruption": corruption,
            "severity": severity,
            "wd_distance": wd_distance
        })


df = pd.DataFrame(df_arr)
df.to_csv('wd_distance.csv')