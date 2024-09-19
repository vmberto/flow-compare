from tensorflow import keras
from keras import layers
import numpy as np
import os
from tensorflow.data import Dataset
import tensorflow as tf
import tensorflow_datasets as tfds
from src.data.data_prep import prepare_cifar10_train_val, comparison_datasets_names, get_cifar10_kfold_splits, prepare_dataset
from src.models.residual_flow import ResidualFlow

output_file = os.path.join(os.getcwd(), "log_likelihood_results.txt")
original_train_ds, original_validation_ds = prepare_cifar10_train_val()
x_train, y_train, x_test, y_test, splits = get_cifar10_kfold_splits(10)

for fold, (train_index, val_index) in splits:
    model = ResidualFlow(num_residual_blocks=12, input_shape=3072)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

    x_train_fold, y_train_fold = x_train[train_index], y_train[train_index]
    x_val_fold, y_val_fold = x_train[val_index], y_train[val_index]

    original_train_ds = prepare_dataset(Dataset.from_tensor_slices((x_train_fold, y_train_fold)))
    original_validation_ds = prepare_dataset(Dataset.from_tensor_slices((x_val_fold, y_val_fold)))

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        min_delta=0.01,
        restore_best_weights=True
    )

    history = model.fit(
        original_train_ds, validation_data=original_validation_ds, batch_size=32, epochs=100,
        callbacks=[early_stopping]
    )

    for corruption_name in comparison_datasets_names:
        corruption_data = tfds.load(f'cifar10_corrupted/{corruption_name}', split='test', as_supervised=True)
        corruption_images = []
        for image, label in corruption_data.take(10000):  # Take 3000 samples
            corruption_images.append(image.numpy())
        corruption_images = np.array(corruption_images).astype("float32") / 255.0
        corruption_images_flattened = tf.reshape(corruption_images, (10000, -1))
        norm = layers.Normalization()
        norm.adapt(corruption_images_flattened)
        normalized_brightness_data = norm(corruption_images_flattened)
        log_likelihoods = model.log_prob(normalized_brightness_data)
        average_log_likelihood = tf.reduce_mean(log_likelihoods).numpy()

        with open(output_file, "a") as f:
            print(f'Writing: Fold {fold + 1}: Average Log-Likelihood on {corruption_name}: {average_log_likelihood}')
            f.write(f"Fold {fold + 1}: Average Log-Likelihood on {corruption_name}: {average_log_likelihood}\n")
