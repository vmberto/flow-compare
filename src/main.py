from lib.data_prep import load_dataset
from lib.flow_model import create_normalizing_flow_maf_model, get_latent_space
from lib.kde_estimation import optimize_bandwidth, calculate_log_likelihood
from lib.latent_space import plot_latent_space
from lib.plot_results import plot_log_likelihoods
from sklearn.neighbors import KernelDensity


def main():
    original_dataset_name = "cifar10"
    comparison_datasets_names = [
        'cifar10',
        'cifar100',
        'cifar10_corrupted/brightness_3',
        'cifar10_corrupted/contrast_3',
        'cifar10_corrupted/defocus_blur_3',
        'cifar10_corrupted/elastic_3',
        'cifar10_corrupted/fog_3',
        'cifar10_corrupted/frost_3',
        'cifar10_corrupted/frosted_glass_blur_3',
        'cifar10_corrupted/gaussian_blur_3',
        'cifar10_corrupted/gaussian_noise_3',
        'cifar10_corrupted/impulse_noise_3',
        'cifar10_corrupted/jpeg_compression_3',
        'cifar10_corrupted/motion_blur_3',
        'cifar10_corrupted/pixelate_3',
        'cifar10_corrupted/saturate_3',
        'cifar10_corrupted/shot_noise_3',
        'cifar10_corrupted/snow_3',
        'cifar10_corrupted/spatter_3',
        'cifar10_corrupted/speckle_noise_3',
        'cifar10_corrupted/zoom_blur_3',
    ]

    flow_model = create_normalizing_flow_maf_model()

    original_dataset = load_dataset(original_dataset_name).take(1)
    original_images, _ = next(iter(original_dataset))
    latents_original = get_latent_space(flow_model, original_images)

    optimal_bandwidth = optimize_bandwidth(latents_original.numpy())
    kde = KernelDensity(bandwidth=optimal_bandwidth)
    kde.fit(latents_original)

    log_likelihoods = []
    all_latents_compared = []
    dataset_names = []

    for dataset_name in comparison_datasets_names:
        try:
            comparison_dataset = load_dataset(dataset_name).take(1)
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            continue
        comparison_images, _ = next(iter(comparison_dataset))
        latents_comparison = get_latent_space(flow_model, comparison_images)
        log_likelihood = calculate_log_likelihood(kde, latents_comparison)
        log_likelihoods.append(log_likelihood)
        dataset_names.append(dataset_name)
        all_latents_compared.append({'latents': latents_comparison, 'dataset': dataset_name})
        print(f"Log Likelihood for {dataset_name}: {log_likelihood}")

    plot_log_likelihoods(dataset_names, log_likelihoods)

    plot_latent_space(latents_original, all_latents_compared, "latent_space_comparison.png")


if __name__ == '__main__':
    main()