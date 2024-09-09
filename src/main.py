from lib.flow_model import get_latent_space
from lib.kde_estimation import optimize_bandwidth, calculate_log_likelihood
from lib.plot_results import plot_similarity
from sklearn.neighbors import KernelDensity
import tensorflow_datasets as tfds
from src.lib.data_prep import prepare_dataset
from src.lib.flow_model import create_normalizing_flow_realnvp_model


def main(original_dataset, comparison_datasets, original_dataset_name, comparison_datasets_names, image_shape):
    # Create the normalizing flow model using the image shape of the datasets
    flow_model = create_normalizing_flow_realnvp_model(image_shape)

    # Pass in the original dataset and calculate its latent space
    original_images, _ = next(iter(original_dataset))
    latents_original = get_latent_space(flow_model, original_images)

    # Fit a KDE to the original dataset's latent space
    optimal_bandwidth = optimize_bandwidth(latents_original.numpy())
    kde = KernelDensity(bandwidth=optimal_bandwidth)
    kde.fit(latents_original)

    # Calculate log likelihood for the original dataset
    log_likelihood_original = calculate_log_likelihood(kde, latents_original)

    # Initialize lists for log likelihoods and dataset names
    log_likelihoods = [log_likelihood_original]
    dataset_names = [original_dataset_name]

    # Process comparison datasets
    for dataset, dataset_name in zip(comparison_datasets, comparison_datasets_names):
        comparison_images, _ = next(iter(dataset))
        latents_comparison = get_latent_space(flow_model, comparison_images)

        # Calculate log likelihood for the comparison dataset
        log_likelihood = calculate_log_likelihood(kde, latents_comparison)
        log_likelihoods.append(log_likelihood)
        dataset_names.append(dataset_name)

        print(f"Log Likelihood for {dataset_name}: {log_likelihood}")

    # Plot similarity percentages based on log likelihoods
    plot_similarity(dataset_names, log_likelihoods, original_dataset_name)


if __name__ == '__main__':
    original_dataset_name = "cats_vs_dogs"
    comparison_datasets_names = ['stanford_dogs']

    # Prepare datasets (ensure they have the same shape)
    original_dataset = prepare_dataset(tfds.load('cats_vs_dogs', split='train', as_supervised=True), image_size=(128, 128))
    comparison_datasets = [prepare_dataset(tfds.load('stanford_dogs', split='train', as_supervised=True), image_size=(128, 128))]

    # Pass the image size dynamically
    main(original_dataset, comparison_datasets, original_dataset_name, comparison_datasets_names, image_shape=(128, 128, 3))
