import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import tensorflow as tf


def optimize_bandwidth(latents):
    latents_np = latents.numpy() if isinstance(latents, tf.Tensor) else latents
    params = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params, cv=5)
    grid.fit(latents_np)
    return grid.best_estimator_.bandwidth


def calculate_log_likelihood(kde, latents):
    return np.sum(kde.score_samples(latents))
