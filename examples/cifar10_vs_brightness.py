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
]

original_dataset = prepare_dataset(tfds.load('cifar10', split='test', as_supervised=True),
                                   image_size=(32, 32))

comparison_datasets = [
    prepare_dataset(tfds.load('cifar10_corrupted/brightness_1', split='test', as_supervised=True), image_size=(32, 32)),
    prepare_dataset(tfds.load('cifar10_corrupted/brightness_2', split='test', as_supervised=True), image_size=(32, 32)),
    prepare_dataset(tfds.load('cifar10_corrupted/brightness_3', split='test', as_supervised=True), image_size=(32, 32)),
    prepare_dataset(tfds.load('cifar10_corrupted/brightness_4', split='test', as_supervised=True), image_size=(32, 32)),
    prepare_dataset(tfds.load('cifar10_corrupted/brightness_5', split='test', as_supervised=True), image_size=(32, 32)),
]

# Pass the image size dynamically
compare_datasets(original_dataset, comparison_datasets, original_dataset_name, comparison_datasets_names,
     image_shape=(32, 32, 3))
