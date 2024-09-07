import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os


def plot_latent_space(latents_original, all_latents_compared, filename):
    plt.figure(figsize=(8, 6))
    tsne = TSNE(n_components=2, perplexity=30)
    pca = PCA(n_components=50)
    latents_tsne_original = tsne.fit_transform(pca.fit_transform(latents_original))
    plt.scatter(latents_tsne_original[:, 0], latents_tsne_original[:, 1], alpha=0.5, label="CIFAR-10")
    for latents_dict in all_latents_compared:
        latents_tsne_compared = tsne.fit_transform(pca.transform(latents_dict['latents']))
        plt.scatter(latents_tsne_compared[:, 0], latents_tsne_compared[:, 1], alpha=0.5, label=latents_dict['dataset'])
    plt.legend()
    plt.title("Latent Space: CIFAR-10 vs Other Datasets")
    output_dir = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Plot saved at: {filepath}")
