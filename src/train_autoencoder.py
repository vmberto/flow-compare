import numpy as np
import tensorflow as tf
from src.data.data_prep import prepare_cifar10_c, get_cifar10_kfold_splits, comparison_datasets_names, get_dataset
from src.lib.kl_divergence import get_kl_divergence
from src.lib.train import train
import pandas as pd
import multiprocessing

from src.lib.wasserstain_distance import get_wasserstein_distance
from src.plots.reconstruction import plot_reconstruction

RECONSTRUCTION_NUMBER = 5
GENERATE_KL_DIVERGENCE = False
PLOT_RECONSTRUCTION = False
GENERATE_WASSERSTEIN_DISTANCE = True


def compute_reconstruction_errors(model, data):
    if isinstance(data, tf.data.Dataset):
        data = np.array([x for x, _ in data])

    reconstructed = model.predict(data)
    errors = np.mean(np.square(data - reconstructed), axis=(1, 2, 3))
    return errors


def main():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_memory_growth(physical_devices[1], True)

    x_train, x_test, splits = get_cifar10_kfold_splits(5)
    input_shape = x_train.shape[1:]

    results = []
    for fold, (train_index, val_index) in splits:
        x_train_fold = x_train[train_index]
        x_val_fold = x_train[val_index]

        train_fold_ds = get_dataset(x_train_fold)
        val_fold_ds = get_dataset(x_val_fold)

        autoencoder, encoder = train(train_fold_ds, val_fold_ds, x_test, input_shape)

        reconstructed_original = autoencoder.predict(x_test[:RECONSTRUCTION_NUMBER])
        latent_original = encoder.predict(x_test)

        for corruption_type in comparison_datasets_names:
            corrupted_ds = prepare_cifar10_c(corruption_type)

            if PLOT_RECONSTRUCTION:
                corrupted_data = np.array([x for x, _ in corrupted_ds.take(RECONSTRUCTION_NUMBER)])
                reconstructed_corrupted = autoencoder.predict(corrupted_data)
                plot_reconstruction(RECONSTRUCTION_NUMBER, reconstructed_corrupted, reconstructed_original, x_test)

            latent_corrupted = encoder.predict(corrupted_ds)

            if GENERATE_KL_DIVERGENCE:
                kldiv_result = get_kl_divergence(corruption_type, fold, latent_original, latent_corrupted)
                results.append(kldiv_result)

            if GENERATE_WASSERSTEIN_DISTANCE:
                wd_result = get_wasserstein_distance(corruption_type, fold, latent_original, latent_corrupted)
                results.append(wd_result)

    pd.DataFrame(results).to_csv('results.csv')


if __name__ == "__main__":
    try:
        p = multiprocessing.Process(target=main)
        p.start()
        p.join()
    except Exception as e:
        print(f"Error occurred: {e}")