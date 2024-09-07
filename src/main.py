from lib.data_prep import get_cifar10, get_svhn, get_fashion_mnist
from lib.flow_model import create_normalizing_flow_model, get_latent_space
from lib.kde_estimation import optimize_bandwidth, calculate_log_likelihood
from lib.latent_space import plot_latent_space
from lib.plot_results import plot_log_likelihoods
from sklearn.neighbors import KernelDensity  # Add this import


def main():
    flow_model = create_normalizing_flow_model()

    cifar10_dataset = get_cifar10()
    cifar_10_images, _ = next(iter(cifar10_dataset))
    latents_original = get_latent_space(flow_model, cifar_10_images)

    optimal_bandwidth = optimize_bandwidth(latents_original.numpy())
    kde = KernelDensity(bandwidth=optimal_bandwidth)
    kde.fit(latents_original)

    log_likelihoods = []
    all_latents_compared = []
    datasets_to_compare = {"SVHN": get_svhn(), "Fashion MNIST": get_fashion_mnist()}  # Replace with other datasets
    dataset_names = []

    for dataset_name, dataset in datasets_to_compare.items():
        dataset_images, _ = next(iter(dataset))
        latents_new_dataset = get_latent_space(flow_model, dataset_images)
        log_likelihood = calculate_log_likelihood(kde, latents_new_dataset)
        log_likelihoods.append(log_likelihood)
        dataset_names.append(dataset_name)
        all_latents_compared.append({'latents': latents_new_dataset, 'dataset': dataset_name})
        print(f"Log Likelihood for {dataset_name}: {log_likelihood}")

    plot_log_likelihoods(dataset_names, log_likelihoods)

    plot_latent_space(latents_original, all_latents_compared, "latent_space_comparison.png")


if __name__ == '__main__':
    main()
