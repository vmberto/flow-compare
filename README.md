# flow-compare
quickly understand the relationship between datasets and detect shifts or anomalies in data distributions

```
 [Start]
    |
    v
[Load Datasets]
    |
    |-- Load the original dataset and the datasets to compare.
    v
[Prepare Data]
    |
    |-- Resize the images, rescale pixel values, and batch the datasets for processing.
    v
[Generate Latent Space using Normalizing Flow]
    |
    |-- Use a normalizing flow model (e.g., RealNVP) to transform the datasets into a lower-dimensional latent space.
    v
[Optimize KDE Bandwidth using Cross-Validation]
    |
    |-- Perform cross-validation to optimize the bandwidth of the Kernel Density Estimator (KDE) for accurate density estimation.
    v
[Fit KDE to Latent Space of Original Dataset]
    |
    |-- Fit the KDE to the latent space of the original dataset, establishing a baseline distribution.
    v
[For each Dataset to Compare]
    |
    |--[Transform Dataset to Latent Space]
    |       |
    |       |-- Transform the comparison dataset into its latent space representation using the normalizing flow model.
    |       v
    |   [Calculate Log Likelihood with KDE]
    |       |
    |       |-- Compute the log likelihood of the comparison dataset using the fitted KDE.
    |       v
    |   [Calculate Similarity as a Percentage]
    |       |
    |       |-- Compute the similarity percentage using the formula:
    |       |       Similarity (%) = (Log Likelihood of Dataset / Log Likelihood of Original Dataset) * 100
    |       v
    |   [Store Similarity Percentage]
    |       |
    |       |-- Save the similarity percentage for later comparison and visualization.
    |
    v
[Visualize Latent Spaces using PCA and t-SNE]
    |
    |-- Apply PCA and t-SNE to visualize the latent spaces of the original and comparison datasets for qualitative comparison.
    v
[Output Results (Print Similarities and Save Plots)]
    |
    |-- Print the calculated similarity percentages and generate plots that visualize both the similarity and latent space differences.
    v
  [End]
```