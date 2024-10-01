from keras.src.applications.xception import Xception
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
import tensorflow_datasets as tfds
from src.data.data_prep import prepare_dataset


def extract_features(dataset):
    """
    Extracts features from a dataset using a pre-trained ResNet50 model.

    Args:
    dataset: A TensorFlow Dataset or numpy array of images to extract features from.

    Returns:
    A numpy array of extracted features.
    """

    # Load the pre-trained ResNet50 model without the top classification layer
    base_model = Xception(weights='imagenet', include_top=False, pooling='avg', input_shape=(71, 71, 3))

    features = base_model.predict(dataset, batch_size=32, verbose=1)

    return features


def compute_wasserstein_distance(features1, features2):
    """
    Computes the Wasserstein Distance between two sets of features.

    Args:
    features1: Feature array from the first dataset (e.g., CIFAR-10).
    features2: Feature array from the second dataset (e.g., CIFAR-10-C).

    Returns:
    The Wasserstein Distance between the two feature sets.
    """

    # Compute Wasserstein distance between each feature dimension
    wd_distances = []
    for i in range(features1.shape[1]):
        wd = wasserstein_distance(features1[:, i], features2[:, i])
        wd_distances.append(wd)

    # Return the average Wasserstein distance across all feature dimensions
    return np.mean(wd_distances)


original_dataset_name = "cifar10"
comparison_datasets_names = [
    'brightness_1',
    'brightness_2',
    'brightness_3',
    'brightness_4',
    'brightness_5',
    'contrast_1',
    'contrast_2',
    'contrast_3',
    'contrast_4',
    'contrast_5',
    'defocus_blur_1',
    'defocus_blur_2',
    'defocus_blur_3',
    'defocus_blur_4',
    'defocus_blur_5',
    'elastic_1',
    'elastic_2',
    'elastic_3',
    'elastic_4',
    'elastic_5',
    'fog_1',
    'fog_2',
    'fog_3',
    'fog_4',
    'fog_5',
    'frost_1',
    'frost_2',
    'frost_3',
    'frost_4',
    'frost_5',
    'frosted_glass_blur_1',
    'frosted_glass_blur_2',
    'frosted_glass_blur_3',
    'frosted_glass_blur_4',
    'frosted_glass_blur_5',
    'gaussian_blur_1',
    'gaussian_blur_2',
    'gaussian_blur_3',
    'gaussian_blur_4',
    'gaussian_blur_5',
    'gaussian_noise_1',
    'gaussian_noise_2',
    'gaussian_noise_3',
    'gaussian_noise_4',
    'gaussian_noise_5',
    'impulse_noise_1',
    'impulse_noise_2',
    'impulse_noise_3',
    'impulse_noise_4',
    'impulse_noise_5',
    'jpeg_compression_1',
    'jpeg_compression_2',
    'jpeg_compression_3',
    'jpeg_compression_4',
    'jpeg_compression_5',
    'motion_blur_1',
    'motion_blur_2',
    'motion_blur_3',
    'motion_blur_4',
    'motion_blur_5',
    'pixelate_1',
    'pixelate_2',
    'pixelate_3',
    'pixelate_4',
    'pixelate_5',
    'saturate_1',
    'saturate_2',
    'saturate_3',
    'saturate_4',
    'saturate_5',
    'shot_noise_1',
    'shot_noise_2',
    'shot_noise_3',
    'shot_noise_4',
    'shot_noise_5',
    'snow_1',
    'snow_2',
    'snow_3',
    'snow_4',
    'snow_5',
    'spatter_1',
    'spatter_2',
    'spatter_3',
    'spatter_4',
    'spatter_5',
    'speckle_noise_1',
    'speckle_noise_2',
    'speckle_noise_3',
    'speckle_noise_4',
    'speckle_noise_5',
    'zoom_blur_1',
    'zoom_blur_2',
    'zoom_blur_3',
    'zoom_blur_4',
    'zoom_blur_5',
]

original_dataset = prepare_dataset(tfds.load('cifar10', split='test', as_supervised=True),
                                   image_size=(71, 71))
comparison_datasets = []
for dataset_name in comparison_datasets_names:
    comparison_datasets.append(prepare_dataset(tfds.load(f'cifar10_corrupted/{dataset_name}', split='test', as_supervised=True), image_size=(71, 71)))

original_features = extract_features(original_dataset)
comparison_features_arr = []
for index, dataset in enumerate(comparison_datasets):
    comparison_features = extract_features(dataset)
    comparison_features_arr.append({"features": comparison_features, "name": comparison_datasets_names[index]})

wd_distances = []
for index, features in enumerate(comparison_features_arr):
    wd_distance = compute_wasserstein_distance(original_features, features["features"])
    wd_distances.append(wd_distance)
    print(f"Wasserstein Distance between {original_dataset_name} and {features['name']}: {wd_distance}")


color_scheme = []
for distance in wd_distances:
    if distance < 0.002:  # Define thresholds for green, yellow, and red
        color_scheme.append('green')
    elif 0.002 <= distance < 0.004:
        color_scheme.append('yellow')
    else:
        color_scheme.append('red')

# Plotting the Wasserstein Distances
plt.figure(figsize=(10, 6))
plt.bar(comparison_datasets_names, wd_distances, color=color_scheme)
plt.title('Wasserstein Distance between Original Dataset and Corrupted Datasets')
plt.xlabel('Corrupted Datasets')
plt.ylabel('Wasserstein Distance')
plt.ylim(0, max(wd_distances) + 0.001)
plt.xticks(rotation=45)
plt.imsave('blabla.png')