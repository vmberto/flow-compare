from scipy.stats import wasserstein_distance


def get_wasserstein_distance(corruption_type, fold, latent_original, latent_corrupted):
    """Compute Wasserstein Distance between latent representations of original and corrupted images."""
    # Flatten the latent spaces for the distance calculation
    latent_original_flat = latent_original.flatten()
    latent_corrupted_flat = latent_corrupted.flatten()

    wd = wasserstein_distance(latent_original_flat, latent_corrupted_flat)
    return {
        'corruption': corruption_type,
        'fold': fold,
        'wasserstein_distance': wd
    }