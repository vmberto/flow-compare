import tensorflow as tf
import tensorflow_datasets as tfds
import keras_cv

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64
INPUT_SHAPE = (32, 32, 3)

# Function to prepare the dataset, resizing images to 32x32x3 and converting grayscale to RGB if necessary
def prepare(ds, shuffle=False, data_augmentation=None):
    resize_and_rescale = tf.keras.Sequential([
        keras_cv.layers.Resizing(INPUT_SHAPE[0], INPUT_SHAPE[1]),  # Resize to 32x32
        keras_cv.layers.Rescaling(1. / 255)  # Rescale pixel values to [0, 1]
    ])

    # Apply resizing and rescaling, convert grayscale images to RGB if necessary
    ds = ds.map(lambda x, y: (tf.image.grayscale_to_rgb(resize_and_rescale(x)) if x.shape[-1] == 1 else resize_and_rescale(x), y),
                num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.batch(BATCH_SIZE)

    if data_augmentation:
        data_augmentation_sequential = tf.keras.Sequential(data_augmentation)
        ds = ds.map(lambda x, y: (data_augmentation_sequential(x, training=True), y), num_parallel_calls=AUTOTUNE)

    return ds.prefetch(buffer_size=AUTOTUNE)


# Load CIFAR-10 dataset
def get_cifar10():
    return prepare(tfds.load("cifar10", split="test", as_supervised=True))


# Load SVHN dataset
def get_svhn():
    return prepare(tfds.load("svhn_cropped", split="test", as_supervised=True))


# Load Fashion MNIST dataset
def get_fashion_mnist():
    return prepare(tfds.load("fashion_mnist", split="test", as_supervised=True))
