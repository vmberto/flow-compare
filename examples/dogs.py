import tensorflow_datasets as tfds
from src.lib.data_prep import prepare_dataset
from src.main import compare_datasets

original_dataset_name = "stanford_dogs"
comparison_datasets_names = [
    'cifar10',
    'cats_vs_dogs',
]

original_dataset = prepare_dataset(tfds.load('stanford_dogs', split='train', as_supervised=True),
                                   image_size=(300, 300))

comparison_datasets = [
    prepare_dataset(tfds.load('cifar10', split='test', as_supervised=True), image_size=(300, 300)),
    prepare_dataset(tfds.load('oxford_iiit_pet', split='train', as_supervised=True), image_size=(300, 300)),
]

compare_datasets(original_dataset, comparison_datasets, original_dataset_name, comparison_datasets_names,
     image_shape=(300, 300, 3))
