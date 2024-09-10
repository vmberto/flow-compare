import tensorflow_datasets as tfds
from src.lib.data_prep import prepare_dataset
from src.main import compare_datasets

original_dataset_name = "oxford_flowers102"
comparison_datasets_names = [
    'tf_flowers',
]

original_dataset = prepare_dataset(tfds.load('oxford_flowers102', split='train', as_supervised=True),
                                   image_size=(128, 128))

comparison_datasets = [
    prepare_dataset(tfds.load('tf_flowers', split='train', as_supervised=True), image_size=(128, 128)),
]

compare_datasets(original_dataset, comparison_datasets, original_dataset_name, comparison_datasets_names,
     image_shape=(128, 128, 3))
