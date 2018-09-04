import data
import model
import numpy as np
import os
import scipy.misc
import tensorflow as tf


MODEL_DIR = 'out' + os.sep + 'model_dir'
LC_OUT_DIR = 'out' + os.sep + 'lc'


def interpolate(a, b, x):
    return (1 - x) * a + x * b


def linear_combinations(a, b, n):
    xs = np.arange(0., 1., 1 / float(n))
    return [interpolate(a, b, x) for x in xs]


def log_csv(m, path):
    a = np.asarray(m)
    np.savetxt(path, a, delimiter=",")


def run_analysis(model_name):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(LC_OUT_DIR):
        os.makedirs(LC_OUT_DIR)

    tf.reset_default_graph()

    # computational graph
    img_batch = tf.placeholder(tf.float32, shape=[None, 28*28], name='img_batch')
    out = model.cnn(img_batch)
    probabilities = out.get('probabilities')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./" + MODEL_DIR + os.sep + "model_" + model_name + ".ckpt")
        a = data.get_class_sample(4)
        b = data.get_class_sample(2)
        result = sess.run(probabilities, feed_dict={
            img_batch: linear_combinations(a, b, 50)
        })
        log_csv(result, './' + LC_OUT_DIR + '/four2two.csv')
        scipy.misc.imsave('./' + LC_OUT_DIR + '/four.png', np.reshape(a, [28, 28]))
        scipy.misc.imsave('./' + LC_OUT_DIR + '/two.png', np.reshape(b, [28, 28]))
        scipy.misc.imsave('./' + LC_OUT_DIR + '/four2two.png', result)
