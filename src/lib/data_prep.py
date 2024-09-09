import tensorflow as tf
import keras_cv


# Example of user-defined dataset preparation
def prepare_dataset(ds, image_size=(128, 128), batch_size=64):
    resize_and_rescale = tf.keras.Sequential([
        keras_cv.layers.Resizing(image_size[0], image_size[1]),
        keras_cv.layers.Rescaling(1.0 / 255)
    ])

    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
