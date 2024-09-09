import tensorflow as tf
import keras
import tensorflow_datasets as tfds
import keras_cv

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64
INPUT_SHAPE = (32, 32, 3)


def prepare(ds, shuffle=False, data_augmentation=None):
    resize_and_rescale = keras.Sequential([
        keras_cv.layers.Resizing(INPUT_SHAPE[0], INPUT_SHAPE[1]),
        keras_cv.layers.Rescaling(1. / 255)
    ])

    ds = ds.map(lambda x, y: (tf.image.grayscale_to_rgb(resize_and_rescale(x)) if x.shape[-1] == 1 else resize_and_rescale(x), y),
                num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)

    return ds.prefetch(buffer_size=AUTOTUNE)


def load_dataset(dataset_name):
    return prepare(tfds.load(dataset_name, split="test", as_supervised=True))