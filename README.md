# flow-compare
quickly understand the relationship between datasets and detect shifts or anomalies in data distributions

```
 [Start]
    |
    v
[Load Datasets]
    |
    v
[Prepare Data (Resizing, Rescaling, Batching)]
    |
    v
[Generate Latent Space using Normalizing Flow]
    |
    v
[Optimize KDE Bandwidth using Cross-Validation]
    |
    v
[Fit KDE to Latent Space of Original CIFAR-10]
    |
    v
[For each Corruption Type]
    |
    |--[Transform Corrupted Images to Latent Space]
    |       |
    |       v
    |   [Calculate Log Likelihood with KDE]
    |       |
    |       v
    |   [Store Log Likelihood]
    |
    v
[Visualize Latent Spaces using PCA and t-SNE]
    |
    v
[Output Results (Print Log Likelihoods and Save Plots)]
    |
    v
  [End]
```