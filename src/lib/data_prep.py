import tensorflow as tf
import keras


# Example of user-defined dataset preparation
def prepare_dataset(ds, image_size=(128, 128), cache=True, repeat=True):
    resize_and_rescale = tf.keras.Sequential([
        tf.keras.layers.Resizing(image_size[0], image_size[1]),
        tf.keras.layers.Rescaling(1. / 255),
    ])

    # Apply resizing and rescaling transformations
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y))

    # Optionally cache the dataset for performance optimization
    if cache:
        ds = ds.cache()

    # Optionally repeat the dataset for multiple epochs
    if repeat:
        ds = ds.repeat()

    # Batch the dataset and prefetch for better performance
    ds = ds.batch(64).prefetch(tf.data.AUTOTUNE)

    return ds
