import tensorflow_datasets as tfds
from src.lib.data_prep import prepare_dataset
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
    'frost_1',
    'frost_2',
    'frost_3',
    'frost_4',
    'frost_5',
    'snow_1',
    'snow_2',
    'snow_3',
    'snow_4',
    'snow_5',
    'motion_blur_1',
    'motion_blur_2',
    'motion_blur_3',
    'motion_blur_4',
    'motion_blur_5',
]

original_dataset = prepare_dataset(tfds.load('cifar10', split='test', as_supervised=True),
                                   image_size=(32, 32))

comparison_datasets = [
    prepare_dataset(tfds.load('cifar10_corrupted/brightness_1', split='test', as_supervised=True), image_size=(32, 32)),
    prepare_dataset(tfds.load('cifar10_corrupted/brightness_2', split='test', as_supervised=True), image_size=(32, 32)),
    prepare_dataset(tfds.load('cifar10_corrupted/brightness_3', split='test', as_supervised=True), image_size=(32, 32)),
    prepare_dataset(tfds.load('cifar10_corrupted/brightness_4', split='test', as_supervised=True), image_size=(32, 32)),
    prepare_dataset(tfds.load('cifar10_corrupted/brightness_5', split='test', as_supervised=True), image_size=(32, 32)),
    prepare_dataset(tfds.load('cifar10_corrupted/contrast_1', split='test', as_supervised=True), image_size=(32,32,3)),
    prepare_dataset(tfds.load('cifar10_corrupted/contrast_2', split='test', as_supervised=True), image_size=(32,32,3)),
    prepare_dataset(tfds.load('cifar10_corrupted/contrast_3', split='test', as_supervised=True), image_size=(32,32,3)),
    prepare_dataset(tfds.load('cifar10_corrupted/contrast_4', split='test', as_supervised=True), image_size=(32,32,3)),
    prepare_dataset(tfds.load('cifar10_corrupted/contrast_5', split='test', as_supervised=True), image_size=(32,32,3)),
    prepare_dataset(tfds.load('cifar10_corrupted/frost_1', split='test', as_supervised=True), image_size=(32,32,3)),
    prepare_dataset(tfds.load('cifar10_corrupted/frost_2', split='test', as_supervised=True), image_size=(32,32,3)),
    prepare_dataset(tfds.load('cifar10_corrupted/frost_3', split='test', as_supervised=True), image_size=(32,32,3)),
    prepare_dataset(tfds.load('cifar10_corrupted/frost_4', split='test', as_supervised=True), image_size=(32,32,3)),
    prepare_dataset(tfds.load('cifar10_corrupted/frost_5', split='test', as_supervised=True), image_size=(32,32,3)),
    prepare_dataset(tfds.load('cifar10_corrupted/snow_1', split='test', as_supervised=True), image_size=(32,32,3)),
    prepare_dataset(tfds.load('cifar10_corrupted/snow_2', split='test', as_supervised=True), image_size=(32,32,3)),
    prepare_dataset(tfds.load('cifar10_corrupted/snow_3', split='test', as_supervised=True), image_size=(32,32,3)),
    prepare_dataset(tfds.load('cifar10_corrupted/snow_4', split='test', as_supervised=True), image_size=(32,32,3)),
    prepare_dataset(tfds.load('cifar10_corrupted/snow_5', split='test', as_supervised=True), image_size=(32,32,3)),
    prepare_dataset(tfds.load('cifar10_corrupted/motion_blur_1', split='test', as_supervised=True), image_size=(32,32,3)),
    prepare_dataset(tfds.load('cifar10_corrupted/motion_blur_2', split='test', as_supervised=True), image_size=(32,32,3)),
    prepare_dataset(tfds.load('cifar10_corrupted/motion_blur_3', split='test', as_supervised=True), image_size=(32,32,3)),
    prepare_dataset(tfds.load('cifar10_corrupted/motion_blur_4', split='test', as_supervised=True), image_size=(32,32,3)),
    prepare_dataset(tfds.load('cifar10_corrupted/motion_blur_5', split='test', as_supervised=True), image_size=(32,32,3)),

]

compare_datasets(original_dataset, comparison_datasets, original_dataset_name, comparison_datasets_names,
     image_shape=(32, 32, 3))
