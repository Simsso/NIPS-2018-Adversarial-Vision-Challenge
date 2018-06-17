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


def fgsm(model_name):
    """
    Loads the given model and computes an adversarial example using FGSM (fast gradient sign method).
    """
    tf.reset_default_graph()

    # computational graph
    img_batch = tf.placeholder(tf.float32, shape=[None, 28 * 28], name='img_batch')
    out = model.cnn(img_batch)
    probabilities = out.get('probabilities')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, pp + 'model_dir' + os.sep + 'model_' + model_name + '.ckpt')
        mnist_class = 0
        a = data.getClassSample(mnist_class)

        # compute partial y / partial x
        # how does the specific class prediction depend on every single input
        gradients = tf.gradients(probabilities[:, mnist_class], img_batch)

        grad_vals = sess.run(gradients, feed_dict={img_batch: [a]})[0]  # unpack batch
        grad_vals = grad_vals[0]  # first class (selection above)
        grad_vals_sign = np.sign(grad_vals) * 1.0/255.  # could be used for the sign method
        scipy.misc.imsave(pp + 'tf_logs/fgsm/grad.png', np.reshape(np.abs(grad_vals), [28, 28]))
        scipy.misc.imsave(pp + 'tf_logs/fgsm/grad_sign.png', np.reshape(grad_vals_sign, [28, 28]))

        # scaling of the attack vector
        epss = np.arange(0., 50., 1)  # epsilon values
        attacks = [a - grad_vals_sign * 1 * x for x in epss]
        attacks = np.clip(attacks, 0, 1)  # clip image pixels to [0,1]

        # compute probabilities for the attack images
        probabilities_out = sess.run(probabilities, feed_dict={img_batch: attacks})

        for i in range(len(probabilities_out)):
            log_output(i, grad_vals_sign * epss[i], attacks[i])

        # log classes
        scipy.misc.imsave(pp + 'tf_logs/fgsm/probabilities.png', probabilities_out)
        print probabilities_out
        print np.argmax(probabilities_out, axis=1)
