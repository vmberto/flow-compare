import re
import numpy as np
from scipy.stats import entropy


def kl_divergence(p, q):
    """ Compute KL Divergence between two probability distributions P and Q """
    # Adding small value to avoid division by zero
    epsilon = 1e-10
    p += epsilon
    q += epsilon
    return entropy(p, q)


def compute_kl_divergence_latent(latent1, latent2, bins=100):
    latent1_flat = latent1.flatten()
    latent2_flat = latent2.flatten()

    latent1_hist, _ = np.histogram(latent1_flat, bins=bins, density=True)
    latent2_hist, _ = np.histogram(latent2_flat, bins=bins, density=True)

    return kl_divergence(latent1_hist, latent2_hist)


def get_kl_divergence(corruption_type, fold, latent_clean, latent_corrupted):
    kl_corrupted = compute_kl_divergence_latent(latent_clean, latent_corrupted)
    print(f'KL Divergence for corrupted {corruption_type.capitalize()}: {kl_corrupted:.4f}')
    pattern = re.compile(r'^(.*)_(\d+)$')
    match = pattern.match(corruption_type)
    corruption = match.group(1)
    severity = match.group(2)

    return {"fold": fold, "corruption": corruption, "severity": severity, "kl_divergence": kl_corrupted}
