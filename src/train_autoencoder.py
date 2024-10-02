import numpy as np
import tensorflow as tf
from scipy.stats import entropy, ttest_ind
from src.data.data_prep import prepare_cifar10_c, get_cifar10_kfold_splits, comparison_datasets_names
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

        autoencoder, encoder = train(x_train_fold, x_val_fold, x_test, input_shape)

        reconstructed_original = autoencoder.predict(x_test[:RECONSTRUCTION_NUMBER])
        latent_original = encoder.predict(x_test)
        # errors_original = compute_reconstruction_errors(autoencoder, x_test)

        for corruption_type in comparison_datasets_names:
            corrupted_images = prepare_cifar10_c(corruption_type)

            # errors_original = compute_reconstruction_errors(autoencoder, corrupted_images)
            #
            # plt.hist(clean_errors, bins=50, alpha=0.5, label='Clean Images')
            # plt.hist(corrupted_errors, bins=50, alpha=0.5, label='Corrupted Images')
            # plt.legend()
            # plt.xlabel('Reconstruction Error')
            # plt.ylabel('Frequency')
            # plt.title(f'Reconstruction Errors: Clean vs {corruption_type.capitalize()}')
            # plt.show()
            #
            # stat, p_value = ttest_ind(clean_errors, corrupted_errors)
            # print(f'T-test statistic: {stat:.4f}, p-value: {p_value:.4e}')

            if PLOT_RECONSTRUCTION:
                reconstructed_corrupted = autoencoder.predict(corrupted_images[:RECONSTRUCTION_NUMBER])
                plot_reconstruction(RECONSTRUCTION_NUMBER, reconstructed_corrupted, reconstructed_original, x_test)

            latent_corrupted = encoder.predict(corrupted_images)

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
    except:
        pass
