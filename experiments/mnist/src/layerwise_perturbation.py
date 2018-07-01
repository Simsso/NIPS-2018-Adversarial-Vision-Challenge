import model
import numpy as np
import os
import tensorflow as tf


MODEL_DIR = 'out' + os.sep + 'model_dir'


def get_perturbation(layers, batch_size):
    layer_perturbations = []
    for layer in layers:
        layer = tf.reshape(layer, [batch_size * 2, -1])  # flatten layer output
        act, act_adv = tf.split(layer, [batch_size, batch_size], axis=0)
        diff = tf.subtract(act, act_adv)
        pert_abs = tf.norm(diff, 'euclidean', axis=1)
        pert_rel = pert_abs / tf.norm(act, 'euclidean', axis=1)
        pert_mean = tf.reduce_mean(pert_rel, axis=0)
        layer_perturbations.append(pert_mean)
    return layer_perturbations


def get_input(xs, xs_pert):
    return np.append(xs, xs_pert, axis=0)


def run_analysis(model_name, xs, xs_adv):
    assert(len(xs) == len(xs_adv))
    sample_count = len(xs)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    tf.reset_default_graph()

    # computational graph
    img_batch = tf.placeholder(tf.float32, shape=[None, 28*28], name='img_batch')
    out = model.cnn(img_batch)
    layers = out.get('layers')
    layerwise_perturbation_mean = get_perturbation(layers, sample_count)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./" + MODEL_DIR + os.sep + "model_" + model_name + ".ckpt")
        layerwise_perturbation_mean_val = sess.run(layerwise_perturbation_mean, feed_dict={
            img_batch: get_input(xs, xs_adv)
        })
        print layerwise_perturbation_mean_val
