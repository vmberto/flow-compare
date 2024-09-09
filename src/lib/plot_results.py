import os
import matplotlib.pyplot as plt
import numpy as np

def plot_similarity(datasets, log_likelihoods, original_dataset_name, errors=None):
    plt.figure(figsize=(14, 9))  # Increase figure size for better clarity

    # Sort datasets and log_likelihoods in ascending order
    sorted_indices = np.argsort(log_likelihoods)
    datasets = np.array(datasets)[sorted_indices]
    log_likelihoods = np.array(log_likelihoods)[sorted_indices]
    if errors is not None:
        errors = np.array(errors)[sorted_indices]

    # Remove the original dataset from the datasets and log likelihoods
    if original_dataset_name in datasets:
        original_dataset_index = np.where(datasets == original_dataset_name)[0][0]
        original_dataset_ll = log_likelihoods[original_dataset_index]  # Get the log likelihood of the original dataset
        datasets = np.delete(datasets, original_dataset_index)
        log_likelihoods = np.delete(log_likelihoods, original_dataset_index)
        if errors is not None:
            errors = np.delete(errors, original_dataset_index)
    else:
        original_dataset_ll = None

    # Calculate similarity as a percentage of the original dataset's log likelihood
    similarities = (log_likelihoods / original_dataset_ll) * 100

    # Color coding: high similarity in green, low similarity in red
    colors = ['#66c2a5' if sim > 50 else '#fc8d62' for sim in similarities]  # Adjust the threshold as needed

    # Increase font size
    plt.rc('font', size=14)
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=14)

    # Plot horizontal bar chart with error bars (excluding the original dataset)
    plt.barh(datasets, similarities, xerr=errors, color=colors, edgecolor='black', alpha=0.7, capsize=5)

    # Annotate bars with similarity values
    for i, v in enumerate(similarities):
        plt.text(v + 1, i, f'{v:.1f}%', color='black', va='center')  # Display percentage with 1 decimal

    # Add a dotted baseline for 100% similarity (original dataset baseline)
    if original_dataset_ll is not None:
        plt.axvline(x=100, color='blue', linestyle='--', label=f'{original_dataset_name} Baseline (100%)')

    # Labels and title with increased font size
    plt.xlabel("Similarity to Original Dataset (%)", fontsize=14)
    plt.ylabel("Dataset", fontsize=14)
    plt.title(f"Similarity of Different Datasets Compared to {original_dataset_name} (Crescent Order)", fontsize=16)
    plt.legend()

    # Save the plot
    plt.tight_layout()
    output_dir = os.path.join(os.getcwd(), 'output')
    filepath = os.path.join(output_dir, f'similarity_plot_sorted_{original_dataset_name}.png')
    plt.savefig(filepath)
    plt.close()

    print(f"Similarity plot saved at {filepath}")