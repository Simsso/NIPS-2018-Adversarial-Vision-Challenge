import data
import model
import numpy as np
import os
import scipy
import tensorflow as tf

pp = './'  # path prefix


def log_output(i, grad, img):
    # scipy.misc.imsave(pp + 'tf_logs/fgsm/grad_%04d.png' % i, np.reshape(grad, [28, 28]))
    scipy.misc.imsave(pp + 'tf_logs/fgsm/img_%04d.png' % i, np.reshape(img, [28, 28]))


def run_fgsm(model_name):
    tf.reset_default_graph()

    # computational graph
    img_batch = tf.placeholder(tf.float32, shape=[None, 28 * 28], name='img_batch')
    out = model.cnn(img_batch)
    probabilities = out.get('probabilities')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, pp + 'model_dir' + os.sep + 'model_' + model_name + '.ckpt')
        mnist_class = 4
        a = data.getClassSample(mnist_class)

        gradients = tf.gradients(probabilities[:, mnist_class], img_batch)

        grad_vals = sess.run(gradients, feed_dict={img_batch: [a]})
        grad_vals = grad_vals[0][0]
        grad_vals_sign = np.sign(grad_vals) * 1e-2
        scipy.misc.imsave(pp + 'tf_logs/fgsm/grad.png', np.reshape(np.abs(grad_vals), [28, 28]))
        scipy.misc.imsave(pp + 'tf_logs/fgsm/grad_sign.png', np.reshape(grad_vals_sign, [28, 28]))

        epss = np.arange(0., 50., 1)  # epsilon values
        attacks = [a + grad_vals * 1 * x for x in epss]
        attacks = np.clip(attacks, 0, 1)

        probabilities_out = sess.run(probabilities, feed_dict={img_batch: attacks})

        for i in range(len(probabilities_out)):
            log_output(i, grad_vals_sign * epss[i], attacks[i])

        scipy.misc.imsave(pp + 'tf_logs/fgsm/probabilities.png', probabilities_out)
