import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow.keras import layers, regularizers

class ResidualFlow(keras.Model):
    def __init__(self, num_residual_blocks, input_shape):
        super(ResidualFlow, self).__init__()

        self.num_residual_blocks = num_residual_blocks
        self.input_shape_ = input_shape

        # Standard Normal distribution for latent space
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(input_shape), scale_diag=tf.ones(input_shape)
        )

        # Create a list of residual blocks
        self.residual_blocks = [ResidualBlock(input_shape) for _ in range(num_residual_blocks)]

        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, x, training=True):
        log_det_jacobian = 0
        for residual_block in self.residual_blocks:
            x, log_det = residual_block(x)
            log_det_jacobian += log_det  # Accumulate log determinant
        return x, log_det_jacobian

    def log_loss(self, x):
        y, logdet = self(x)
        log_likelihood = self.distribution.log_prob(y) + logdet
        return -tf.reduce_mean(log_likelihood)

    def train_step(self, data):
        # If `data` is a dataset, unpack it
        if isinstance(data, tuple):
            x, _ = data  # Assumes (features, labels) but you may not have labels
        else:
            x = data

        with tf.GradientTape() as tape:
            loss = self.log_loss(x)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        # If `data` is a dataset, unpack it
        if isinstance(data, tuple):
            x, _ = data  # Assumes (features, labels)
        else:
            x = data

        loss = self.log_loss(x)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def log_prob(self, x):
        z, log_det_jacobian = self.call(x)
        log_prob_z = self.distribution.log_prob(z)
        return log_prob_z + log_det_jacobian


class ResidualBlock(layers.Layer):
    def __init__(self, input_shape, hidden_dim=256, reg=0.01):
        super(ResidualBlock, self).__init__()
        self.input_shape_ = input_shape
        self.hidden_dim = hidden_dim

        # Define layers for residual block network
        self.dense1 = layers.Dense(hidden_dim, activation='relu', kernel_regularizer=regularizers.l2(reg))
        self.dense2 = layers.Dense(hidden_dim, activation='relu', kernel_regularizer=regularizers.l2(reg))
        self.dense3 = layers.Dense(hidden_dim, activation='relu', kernel_regularizer=regularizers.l2(reg))

        # Final layer outputs the residual f(x)
        self.residual = layers.Dense(input_shape, activation='linear', kernel_regularizer=regularizers.l2(reg))

    def call(self, x):
        # Residual network transforms f(x)
        h = self.dense1(x)
        h = self.dense2(h)
        h = self.dense3(h)
        residual = self.residual(h)

        # Apply the residual transformation: z = x + f(x)
        z = x + residual

        # For residual flows, the Jacobian is approximately the identity matrix, so log-det is 0
        log_det_jacobian = 0

        return z, log_det_jacobian