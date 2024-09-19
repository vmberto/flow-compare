import tensorflow_datasets as tfds
from src.data.data_prep import prepare_dataset
from src.main import compare_datasets

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


# Load CIFAR-10 'train' dataset and split into training and validation sets
original_dataset = tfds.load('cifar10', split='train', as_supervised=True)

# Split into training and validation (90% train, 10% validation)
train_size = 0.9
train_ds = original_dataset.take(int(train_size * len(original_dataset)))  # 90% for training
validation_ds = original_dataset.skip(int(train_size * len(original_dataset)))  # 10% for validation

# Prepare the datasets
original_train_ds = prepare_dataset(train_ds, image_size=(32, 32), cache=True, repeat=True)
original_validation_ds = prepare_dataset(validation_ds, image_size=(32, 32), cache=True, repeat=False)



comparison_datasets = []
for dataset_name in comparison_datasets_names:
    comparison_datasets.append(prepare_dataset(tfds.load(f'cifar10_corrupted/{dataset_name}', split='test', as_supervised=True), image_size=(32, 32)))


compare_datasets(original_train_ds, original_validation_ds, comparison_datasets, original_dataset_name, comparison_datasets_names, image_shape=(32, 32, 3))
