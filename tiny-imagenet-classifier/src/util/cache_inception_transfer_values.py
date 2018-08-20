import tensorflow as tf
import model.inception_v3 as inception_v3
import data.tiny_imagenet as data
import pickle 
import numpy as np
import os
import sys

slim = tf.contrib.slim

ACTIVATION_DIM = 2048

##################### Tiny ImageNet Loading Helper ######################
import cv2

# Helper Functions to load all the images, not create a `tf.train.input_producer`-queue
def load_image(filename):
    img = cv2.imread(filename)
    return img

def load_tiny_image_net(mode, limit=None):
    filenames_labels = data.load_filenames_labels(mode)
    # import: don't shuffle here, otherwise the cached transfer values will be useless!
    if limit:
        filenames_labels = filenames_labels[:limit]
    images = np.array([load_image(img) for img, _ in filenames_labels])
    labels = np.array([label for _, label in filenames_labels])
    labels = labels.astype(np.uint8)
    return images, labels
#########################################################################

def inception_v3_features(images):
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        _, endpoints = inception_v3.inception_v3(
            inputs=images,
            num_classes=1001,
            is_training=False,
            create_aux_logits=False)
    features = endpoints['PreLogits']   # Nx1x1xACTIVATION_DIM
    return features


def run_batch(batch):
    scaled_batch = scaled_images = tf.image.resize_images(images=batch, size=[299, 299])
    activations = inception_v3_features(scaled_batch)
    return activations


def inference_in_batches(all_images, batch_size):
    """Returns a numpy array of the activations of shape len(all_images)x2048
    - all_images: 2d numpy array of shape Nx(data.IMG_DIM)x(data.IMG_DIM)x(data.IMG_CHANNELS)
    """
    graph = tf.Graph()
    with graph.as_default():
        batch_placeholder = tf.placeholder(tf.float32, shape=[None, data.IMG_DIM, data.IMG_DIM, data.IMG_CHANNELS])
        activations = run_batch(batch=batch_placeholder)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = inception_v3.create_saver()

    sess = tf.Session(graph=graph)
    num_images = len(all_images)
    result = np.ndarray(shape=(num_images, ACTIVATION_DIM))
    
    with sess.as_default():
        sess.run(init)
        inception_v3.restore(sess, saver)

        full_batches = num_images // batch_size
        num_batches = full_batches if num_images % batch_size == 0 else full_batches + 1

        for i in range(num_batches):
            msg = "\r- Processing batch: {0:>6} / {1}".format(i+1, num_batches)
            sys.stdout.write(msg)
            sys.stdout.flush()

            from_idx = i*batch_size
            to_idx = min((i+1)*batch_size, num_images)
            batch_values = all_images[from_idx:to_idx]
            batch_result = sess.run(activations, feed_dict={
                batch_placeholder: batch_values
            })

            result[from_idx:to_idx] = np.squeeze(batch_result)  # remove 1-d dimensions
        print("") # new line

    return result


def read_cache_or_generate_activations(cache_path, all_images, batch_size=64):
    # If the cache-file exists
    if os.path.exists(cache_path):
        # Load the cached data from the file
        with open(cache_path, mode='rb') as file:
            activations = pickle.load(file)
        print("- Activations loaded from cache-file: " + cache_path)
    else:
        # The cache-file does not exist.
        # Generate activations
        activations = inference_in_batches(all_images, batch_size=batch_size)

        # Save the data to a cache-file.
        with open(cache_path, mode='wb') as file:
            pickle.dump(activations, file)
        print("- Activations saved to cache-file: " + cache_path)
    return activations


CACHE_DIR = os.path.expanduser("~/.models/cached_activations/")

def get_cache_path(mode):
    return os.path.join(CACHE_DIR, "tiny_imagenet_" + mode + ".pkl")
