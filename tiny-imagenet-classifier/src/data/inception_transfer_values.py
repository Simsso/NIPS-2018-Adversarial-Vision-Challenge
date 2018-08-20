import util.cache_inception_transfer_values as cache
import os
import numpy as np
import tensorflow as tf
import data.tiny_imagenet as tiny_imagenet

ACTIVATION_DIM = cache.ACTIVATION_DIM
NUM_VALIDATION_SAMPLES = tiny_imagenet.NUM_VALIDATION_SAMPLES
NUM_TRAIN_SAMPLES = tiny_imagenet.NUM_TRAIN_SAMPLES


def get_activations_labels(mode):
    images, labels = cache.load_tiny_image_net(mode=mode)
    activations = cache.read_cache_or_generate_activations(cache_path=cache.get_cache_path(mode), all_images=images, batch_size=500)

    # shuffle activations and labels (important: only *after* the activation generation!)
    idx = np.random.permutation(len(labels))
    labels = labels[idx]
    activations = activations[idx]

    return activations, labels