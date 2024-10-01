import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import entropy
from src.data.data_prep import prepare_cifar10_c, prepare_cifar10, comparison_datasets_names
from src.lib.train import train
import re
import tensorflow as tf
import pandas as pd
import multiprocessing


def compute_reconstruction_errors(model, data):
    reconstructed = model.predict(data)
    errors = np.mean(np.square(data - reconstructed), axis=(1, 2, 3))
    return errors


def main():
    n = 5

    x_train, x_val, x_test = prepare_cifar10()
    input_shape = x_train.shape[1:]
    autoencoder, encoder = train(x_train, x_val, x_test, input_shape)
    clean_errors = compute_reconstruction_errors(autoencoder, x_test)
    latent_clean = encoder.predict(x_test[:n])

    kl_results = []

    for corruption_type in comparison_datasets_names:
        corrupted_images = prepare_cifar10_c(corruption_type)

        corrupted_errors = compute_reconstruction_errors(autoencoder, corrupted_images)

        # plt.hist(clean_errors, bins=50, alpha=0.5, label='Clean Images')
        # plt.hist(corrupted_errors, bins=50, alpha=0.5, label='Corrupted Images')
        # plt.legend()
        # plt.xlabel('Reconstruction Error')
        # plt.ylabel('Frequency')
        # plt.title(f'Reconstruction Errors: Clean vs {corruption_type.capitalize()}')
        # plt.show()

        # Statistical analysis
        stat, p_value = ttest_ind(clean_errors, corrupted_errors)
        print(f'T-test statistic: {stat:.4f}, p-value: {p_value:.4e}')

        # Visualize original, reconstructed, corrupted, and difference images
        plt.figure(figsize=(15, 12))  # Adjusted figure size to accommodate 4 rows

        # Get reconstructed images for original and corrupted images
        reconstructed_original = autoencoder.predict(x_test[:n])
        reconstructed_corrupted = autoencoder.predict(corrupted_images[:n])


        # for i in range(n):
        #     # Original Images
        #     ax = plt.subplot(4, n, i + 1)
        #     plt.imshow(x_test[i])
        #     plt.title("Original")
        #     plt.axis('off')
        #
        #     # Reconstructed Original Images
        #     ax = plt.subplot(4, n, i + 1 + n)
        #     plt.imshow(reconstructed_original[i])
        #     plt.title("Reconstructed Original")
        #     plt.axis('off')
        #
        #     # Corrupted Images
        #     ax = plt.subplot(4, n, i + 1 + 2 * n)
        #     plt.imshow(reconstructed_corrupted[i])
        #     plt.title("Corrupted")
        #     plt.axis('off')
        #
        #     # Difference between Corrupted and Reconstructed Corrupted Images
        #     ax = plt.subplot(4, n, i + 1 + 3 * n)
        #     difference = np.abs(reconstructed_original[i] - reconstructed_corrupted[i])
        #     plt.imshow(difference)
        #     plt.title("Difference")
        #     plt.axis('off')
        #
        # plt.tight_layout()
        # plt.show()


        def kl_divergence(p, q):
            """ Compute KL Divergence between two probability distributions P and Q """
            # Adding small value to avoid division by zero
            epsilon = 1e-10
            p += epsilon
            q += epsilon
            return entropy(p, q)


        def compute_kl_divergence_latent(latent1, latent2, bins=100):
            # Flatten the latent vectors
            latent1_flat = latent1.flatten()
            latent2_flat = latent2.flatten()

            # Generate histograms (with normalization to form probability distributions)
            latent1_hist, _ = np.histogram(latent1_flat, bins=bins, density=True)
            latent2_hist, _ = np.histogram(latent2_flat, bins=bins, density=True)

            # Compute KL divergence between the latent space distributions
            return kl_divergence(latent1_hist, latent2_hist)


        # Apply to your images
        latent_corrupted = encoder.predict(corrupted_images[:n])
        kl_corrupted = compute_kl_divergence_latent(latent_clean, latent_corrupted)
        print(f'KL Divergence for corrupted {corruption_type.capitalize()}: {kl_corrupted:.4f}')
        pattern = re.compile(r'^(.*)_(\d+)$')
        match = pattern.match(corruption_type)
        corruption = match.group(1)
        severity = match.group(2)

        kl_results.append({"fold": fold, "corruption": corruption, "severity": severity, "kl_divergence": kl_corrupted})


    pd.DataFrame(kl_results).to_csv('results.csv')
        #
        # # Flatten the latent representations if necessary (depends on your model)
        # latent_clean_flat = latent_clean.reshape(latent_clean.shape[0], -1)
        # latent_corrupted_flat = latent_corrupted.reshape(latent_corrupted.shape[0], -1)
        #
        # # Dimensionality reduction using PCA (you can also use t-SNE for non-linear reduction)
        # pca = PCA(n_components=2)
        # latent_clean_2d = pca.fit_transform(latent_clean_flat)
        # latent_corrupted_2d = pca.fit_transform(latent_corrupted_flat)
        #
        # # Plot the latent space
        # plt.figure(figsize=(8, 6))
        # plt.scatter(latent_clean_2d[:, 0], latent_clean_2d[:, 1], label='Clean Latent Space', alpha=0.6, color='blue')
        # plt.scatter(latent_corrupted_2d[:, 0], latent_corrupted_2d[:, 1], label=f'Corrupted Latent Space', alpha=0.6, color='red')
        # plt.legend()
        # plt.title('Latent Space Visualization (2D)')
        # plt.xlabel('Latent Dimension 1')
        # plt.ylabel('Latent Dimension 2')
        # plt.show()


if __name__ == "__main__":
    try:
        p = multiprocessing.Process(target=main)
        p.start()
        p.join()
    except:
        pass
