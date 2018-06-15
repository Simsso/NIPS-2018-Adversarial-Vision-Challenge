import data
import model
import numpy as np
import os
import scipy
import tensorflow as tf


def run_fgsm(model_name):
    tf.reset_default_graph()

    # computational graph
    img_batch = tf.placeholder(tf.float32, shape=[None, 28 * 28], name='img_batch')
    out = model.cnn(img_batch)
    probabilities = out.get('probabilities')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model_dir" + os.sep + "model_" + model_name + ".ckpt")
        a = data.getClassSample(4)

        gradients = tf.gradients(probabilities, img_batch)

        grad_vals = sess.run(gradients, feed_dict={img_batch: [a]})[0]

        scipy.misc.imsave('./tf_logs/fgsm/grad.png', np.reshape(grad_vals[0], [28, 28]))
