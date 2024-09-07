import tensorflow as tf
import tensorflow_datasets as tfds
import keras_cv

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64
INPUT_SHAPE = (32, 32, 3)


def prepare(ds, shuffle=False, data_augmentation=None):
    resize_and_rescale = tf.keras.Sequential([
        keras_cv.layers.Resizing(INPUT_SHAPE[0], INPUT_SHAPE[1]),
        keras_cv.layers.Rescaling(1. / 255)
    ])

    ds = ds.map(lambda x, y: (tf.image.grayscale_to_rgb(resize_and_rescale(x)) if x.shape[-1] == 1 else resize_and_rescale(x), y),
                num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.batch(BATCH_SIZE)

    if data_augmentation:
        data_augmentation_sequential = tf.keras.Sequential(data_augmentation)
        ds = ds.map(lambda x, y: (data_augmentation_sequential(x, training=True), y), num_parallel_calls=AUTOTUNE)

    return ds.prefetch(buffer_size=AUTOTUNE)


def load_dataset(dataset_name):
    return prepare(tfds.load(dataset_name, split="test", as_supervised=True))