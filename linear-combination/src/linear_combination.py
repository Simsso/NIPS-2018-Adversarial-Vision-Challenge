import data
import model
import numpy as np
import os
import scipy.misc
import tensorflow as tf

BATCH_SIZE = 100


def interpolate(a, b, x):
    return (1 - x) * a + x * b


def linear_combinations(a, b, n):
    xs = np.arange(0., 1., 1 / float(n))
    return [interpolate(a, b, x) for x in xs]


def log_csv(m, path):
    a = np.asarray(m)
    np.savetxt(path, a, delimiter=",")


def run_analysis(model_name):
    tf.reset_default_graph()

    # computational graph
    img_batch = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 28*28], name='img_batch')
    out = model.cnn(img_batch)
    probabilities = out.get('probabilities')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model_dir" + os.sep + "model_" + model_name + ".ckpt")
        a = data.getClassSample(4)
        b = data.getClassSample(2)
        result = sess.run(probabilities, feed_dict={
            img_batch: linear_combinations(a, b, BATCH_SIZE)
        })
        log_csv(result, './tf_logs/four2two.csv')
        scipy.misc.imsave('./tf_logs/four.png', np.reshape(a, [28, 28]))
        scipy.misc.imsave('./tf_logs/two.png', np.reshape(b, [28, 28]))
        scipy.misc.imsave('./tf_logs/four2two.png', result)
