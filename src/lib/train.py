import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks
import numpy as np
import tensorflow as tf


def train(train_ds, val_ds, x_test, input_shape):
    # Define a residual block
    def residual_block(x, filters, kernel_size=3, stride=1):
        shortcut = x
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)

        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        x = layers.add([shortcut, x])
        x = layers.Activation('relu')(x)
        return x

    # Encoder
    encoder_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), strides=2, padding='same')(encoder_input)  # 16x16x64
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same')(x)  # 8x8x128
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = layers.Conv2D(256, (3, 3), strides=2, padding='same')(x)  # 4x4x256
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    encoder_output = x
    encoder = models.Model(encoder_input, encoder_output, name='encoder')

    # Decoder
    decoder_input = layers.Input(shape=encoder_output.shape[1:])
    x = residual_block(decoder_input, 256)
    x = residual_block(x, 256)
    x = layers.Conv2DTranspose(256, (3, 3), strides=2, padding='same')(x)  # 8x8x256
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x)  # 16x16x128
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)  # 32x32x64
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Output layer with sigmoid activation
    decoder_output = x
    decoder = models.Model(decoder_input, decoder_output, name='decoder')

    # Autoencoder model
    autoencoder_input = encoder_input
    encoded_img = encoder(autoencoder_input)
    decoded_img = decoder(encoded_img)
    autoencoder = models.Model(autoencoder_input, decoded_img, name='autoencoder')

    # Compile model
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Training
    epochs = 50
    batch_size = 256
    history = autoencoder.fit(
        train_ds,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=val_ds,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        ]
    )

    return autoencoder, encoder