import tensorflow as tf
import keras


# Example of user-defined dataset preparation
def prepare_dataset(ds, image_size=(128, 128)):
    resize_and_rescale = keras.Sequential([
        keras.layers.Resizing(image_size[0], image_size[1]),
        keras.layers.Rescaling(1./255),
    ])
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y))
    ds = ds.batch(64).prefetch(tf.data.AUTOTUNE)
    return ds
