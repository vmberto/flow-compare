import tensorflow as tf
import keras
import numpy as np
import tensorflow_datasets as tfds
from sklearn.model_selection import KFold


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



def prepare_dataset(ds, image_size=(32, 32), cache=True, repeat=False, preprocess_layers=None):
    resize_and_rescale = keras.models.Sequential([
        keras.layers.Resizing(image_size[0], image_size[1]),  # Resize to 32x32
        keras.layers.Rescaling(1. / 255),  # Normalize pixel values to [0, 1]
    ])

    ds = ds.map(lambda x, y: (resize_and_rescale(x), y))

    if repeat:
        ds = ds.repeat()

    if preprocess_layers:
        ds = ds.map(lambda x, y: (preprocess_layers(x), y))

    ds = ds.batch(128).prefetch(tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()

    return ds


# Load CIFAR-10 dataset
def prepare_cifar10():
    dataset_train = tfds.load('cifar10', split='train', as_supervised=True)
    dataset_test = tfds.load('cifar10', split='test', as_supervised=True)

    x_train = np.array([image for image, label in tfds.as_numpy(dataset_train)])
    x_test = np.array([image for image, label in tfds.as_numpy(dataset_test)])

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return x_train, x_test


def prepare_cifar10_c(corruption_type):
    dataset = tfds.load(f'cifar10_corrupted/{corruption_type}', split='test', as_supervised=True)

    x_corrupted = np.array([image for image, label in tfds.as_numpy(dataset)])

    x_corrupted = x_corrupted.astype('float32') / 255.0

    corrupted_ds = prepare_dataset(tf.data.Dataset.from_tensor_slices((x_corrupted, x_corrupted)))

    return corrupted_ds


def get_cifar10_kfold_splits(n_splits):
    x_train, x_test = prepare_cifar10()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    dataset_splits = list(enumerate(kf.split(x_train)))

    return x_train, x_test, dataset_splits


def get_dataset(x_data):
    return prepare_dataset(tf.data.Dataset.from_tensor_slices((x_data, x_data)))
