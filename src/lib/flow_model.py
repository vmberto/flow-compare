import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


def create_normalizing_flow_model():
    base_distribution = tfd.MultivariateNormalDiag(loc=tf.zeros([32 * 32 * 3]), scale_diag=tf.ones([32 * 32 * 3]))
    bijectors = []
    for _ in range(4):
        bijectors.append(tfb.RealNVP(num_masked=32 * 32 * 3 // 2,
                                     shift_and_log_scale_fn=tfp.bijectors.real_nvp_default_template(
                                         hidden_layers=[256, 256])))
        bijectors.append(tfb.Permute(permutation=tf.range(32 * 32 * 3 - 1, -1, -1)))
    flow_bijector = tfb.Chain(list(reversed(bijectors)))
    return tfd.TransformedDistribution(distribution=base_distribution, bijector=flow_bijector)


def get_latent_space(flow_model, images):
    return flow_model.bijector.forward(tf.reshape(images, [images.shape[0], -1]))
