import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

def create_normalizing_flow_maf_model():
    base_distribution = tfd.MultivariateNormalDiag(
        loc=tf.zeros([32 * 32 * 3]),
        scale_diag=tf.ones([32 * 32 * 3])
    )

    bijectors = []
    for _ in range(4):
        bijectors.append(tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
                params=2,  # Shift and log scale
                hidden_units=[256, 256],
                event_shape=[32 * 32 * 3]
            )
        ))
        bijectors.append(tfb.Permute(permutation=tf.range(32 * 32 * 3 - 1, -1, -1)))

    flow_bijector = tfb.Chain(list(reversed(bijectors)))

    return tfd.TransformedDistribution(distribution=base_distribution, bijector=flow_bijector)


import tensorflow_probability as tfp
import tensorflow as tf


def create_normalizing_flow_realnvp_model(image_shape):
    """Create a RealNVP normalizing flow model that adapts to the input image shape."""
    num_pixels = tf.reduce_prod(image_shape)  # Dynamically calculate the number of pixels (e.g., 32*32*3 or 128*128*3)

    base_distribution = tfp.distributions.MultivariateNormalDiag(
        loc=tf.zeros([num_pixels]), scale_diag=tf.ones([num_pixels])
    )

    num_bijectors = 4
    bijectors = []

    for i in range(num_bijectors):
        bijectors.append(tfp.bijectors.RealNVP(
            num_masked=num_pixels // 2,
            shift_and_log_scale_fn=tfp.bijectors.real_nvp_default_template(hidden_layers=[256, 256])
        ))
        bijectors.append(tfp.bijectors.Permute(permutation=[i for i in reversed(range(num_pixels))]))

    flow_bijector = tfp.bijectors.Chain(list(reversed(bijectors)))

    flow_model = tfp.distributions.TransformedDistribution(
        distribution=base_distribution,
        bijector=flow_bijector
    )

    return flow_model

def get_latent_space(flow_model, images):
    return flow_model.bijector.forward(tf.reshape(images, [images.shape[0], -1]))
