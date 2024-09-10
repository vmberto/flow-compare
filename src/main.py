from src.lib.flow_model import create_normalizing_flow_realnvp_model, get_latent_space, train_realnvp
from src.lib.kde_estimation import optimize_bandwidth, calculate_log_likelihood
from src.lib.plot_results import plot_similarity
from src.lib.latent_space import plot_latent_space
from sklearn.neighbors import KernelDensity


def compare_datasets(original_dataset, comparison_datasets, original_dataset_name, comparison_datasets_names, image_shape):
    flow_model = create_normalizing_flow_realnvp_model(image_shape)
    train_realnvp(flow_model, flow_model.bijector.bijectors, original_dataset, 1, .0001)

    original_images, _ = next(iter(original_dataset))
    latents_original = get_latent_space(flow_model, original_images)

    optimal_bandwidth = optimize_bandwidth(latents_original.numpy())
    kde_original = KernelDensity(bandwidth=optimal_bandwidth)
    kde_original.fit(latents_original)

    log_likelihood_original = calculate_log_likelihood(kde_original, latents_original)

    log_likelihoods = [log_likelihood_original]
    dataset_names = [original_dataset_name]
    all_latents_compared = []

    # Process comparison datasets
    for dataset, dataset_name in zip(comparison_datasets, comparison_datasets_names):
        comparison_images, _ = next(iter(dataset))
        latents_comparison = get_latent_space(flow_model, comparison_images)

        log_likelihood = calculate_log_likelihood(kde_original, latents_comparison)
        log_likelihoods.append(log_likelihood)

        dataset_names.append(dataset_name)
        all_latents_compared.append({'latents': latents_comparison, 'dataset': dataset_name})

        print(f"Log Likelihood for {dataset_name}: {log_likelihood}")

    plot_similarity(dataset_names, log_likelihoods, original_dataset_name)

    plot_latent_space(latents_original, all_latents_compared, "latent_space_comparison.png", original_dataset_name)
