import matplotlib.pyplot as plt
import numpy as np


def plot_reconstruction(n, reconstructed_corrupted, reconstructed_original, x_test):
    plt.figure(figsize=(15, 12))
    for i in range(n):
        _ = plt.subplot(4, n, i + 1)
        plt.imshow(x_test[i])
        plt.title("Original")
        plt.axis('off')

        _ = plt.subplot(4, n, i + 1 + n)
        plt.imshow(reconstructed_original[i])
        plt.title("Rec. Original")
        plt.axis('off')

        _ = plt.subplot(4, n, i + 1 + 2 * n)
        plt.imshow(reconstructed_corrupted[i])
        plt.title("Rec. Corrupted")
        plt.axis('off')

        _ = plt.subplot(4, n, i + 1 + 3 * n)
        difference = np.abs(reconstructed_original[i] - reconstructed_corrupted[i])
        plt.imshow(difference)
        plt.title("Difference")
        plt.axis('off')
    plt.tight_layout()
    plt.show()