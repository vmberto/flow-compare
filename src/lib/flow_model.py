import tensorflow as tf
import tensorflow_probability as tfp
import keras

tfb = tfp.bijectors
tfd = tfp.distributions


class RealNVPBijector(tfb.Bijector):
    def __init__(self, num_masked, hidden_units):
        super(RealNVPBijector, self).__init__(forward_min_event_ndims=1)
        self.num_masked = num_masked

        # Define the shift and log_scale networks
        self.shift_net = keras.Sequential([
            keras.layers.InputLayer(shape=[self.num_masked]),
            keras.layers.Dense(hidden_units, activation='relu'),
            keras.layers.Dense(self.num_masked)
        ])

        self.log_scale_net = keras.Sequential([
            keras.layers.InputLayer(shape=[self.num_masked]),
            keras.layers.Dense(hidden_units, activation='relu'),
            keras.layers.Dense(self.num_masked)
        ])

    def _forward(self, x):
        x1, x2 = x[..., :self.num_masked], x[..., self.num_masked:]
        shift = self.shift_net(x1)
        log_scale = self.log_scale_net(x1)
        y2 = x2 * tf.exp(log_scale) + shift
        return tf.concat([x1, y2], axis=-1)

    def _inverse(self, y):
        y1, y2 = y[..., :self.num_masked], y[..., self.num_masked:]
        shift = self.shift_net(y1)
        log_scale = self.log_scale_net(y1)
        x2 = (y2 - shift) * tf.exp(-log_scale)
        return tf.concat([y1, x2], axis=-1)

    def _forward_log_det_jacobian(self, x):
        x1 = x[..., :self.num_masked]
        log_scale = self.log_scale_net(x1)
        return tf.reduce_sum(log_scale, axis=-1)

    def _inverse_log_det_jacobian(self, y):
        y1 = y[..., :self.num_masked]
        log_scale = self.log_scale_net(y1)
        return -tf.reduce_sum(log_scale, axis=-1)


# Create RealNVP model for images
def create_normalizing_flow_realnvp_model(image_shape, hidden_units=256):
    """Create a RealNVP normalizing flow model that adapts to the input image shape."""
    num_pixels = int(tf.reduce_prod(image_shape).numpy())  # Calculate the number of pixels in the image

    # Base distribution: Multivariate Normal with diagonal covariance
    base_distribution = tfd.MultivariateNormalDiag(
        loc=tf.zeros([num_pixels]), scale_diag=tf.ones([num_pixels])
    )

    # Create a list of bijectors
    num_bijectors = 4
    bijectors = []
    for _ in range(num_bijectors):
        bijectors.append(RealNVPBijector(num_masked=num_pixels // 2, hidden_units=hidden_units))
        # Permute between each layer
        bijectors.append(tfb.Permute(permutation=list(reversed(range(num_pixels)))))

    # Chain the bijectors and form the transformed distribution
    flow_bijector = tfb.Chain(bijectors)
    flow_model = tfd.TransformedDistribution(distribution=base_distribution, bijector=flow_bijector)

    return flow_model


# Process the image data and get latent space
def get_latent_space(flow_model, images):
    """Flatten the images and pass through the bijector to get the latent space."""
    flattened_images = tf.reshape(images, [images.shape[0], -1])
    return flow_model.bijector.forward(flattened_images)


# Loss function
def compute_loss(flow_model, images):
    flattened_images = tf.reshape(images, [images.shape[0], -1])
    return -tf.reduce_mean(flow_model.log_prob(flattened_images))


def train_realnvp(flow_model, bijectors, dataset, epochs=10, learning_rate=1e-3):
    optimizer = keras.optimizers.Adam(learning_rate)

    trainable_variables = []
    for bijector in bijectors:
        if isinstance(bijector, RealNVPBijector):
            trainable_variables.extend(bijector.shift_net.trainable_variables)
            trainable_variables.extend(bijector.log_scale_net.trainable_variables)

    if not trainable_variables:
        raise ValueError("No trainable variables found.")

    for epoch in range(epochs):
        for step, (images, _) in enumerate(dataset):
            with tf.GradientTape() as tape:
                loss = compute_loss(flow_model, images)
            gradients = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))

            if step % 100 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.numpy()}")

    print("Training complete.")
