# model.py
from tensorflow.keras import layers, models


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


# Build the encoder model
def build_encoder(input_shape):
    encoder_input = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (3, 3), strides=2, padding='same')(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = layers.Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = residual_block(x, 128)
    x = residual_block(x, 128)

    x = layers.Conv2D(256, (3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = residual_block(x, 256)
    x = residual_block(x, 256)

    encoder_output = x
    encoder = models.Model(encoder_input, encoder_output, name='encoder')
    return encoder


# Build the decoder model
def build_decoder(encoder_output_shape):
    decoder_input = layers.Input(shape=encoder_output_shape)

    x = residual_block(decoder_input, 256)
    x = residual_block(x, 256)

    x = layers.Conv2DTranspose(256, (3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = residual_block(x, 128)
    x = residual_block(x, 128)

    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    decoder_output = x

    decoder = models.Model(decoder_input, decoder_output, name='decoder')
    return decoder