import matplotlib.pyplot as plt
import os


def plot_log_likelihoods(datasets, log_likelihoods):
    plt.figure(figsize=(10, 6))
    plt.barh(datasets, log_likelihoods, color='skyblue')
    plt.xlabel("Log Likelihood (derived from PDF)")
    plt.ylabel("Dataset")
    plt.title("Log Likelihoods of Different Datasets Compared to CIFAR-10")
    plt.tight_layout()
    output_dir = os.path.join(os.getcwd(), 'output')
    filepath = os.path.join(output_dir, 'log_likelihoods_plot.png')
    plt.savefig(filepath)
    plt.close()
    print("Log Likelihoods plot saved as log_likelihoods_plot.png")
