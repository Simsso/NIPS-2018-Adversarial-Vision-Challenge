import data
import model
import numpy as np
import os
import scipy.misc
import tensorflow as tf

pp = './'  # path prefix
FGSM_DIR = 'out' + os.sep + 'fgsm'
MODEL_DIR = 'out' + os.sep + 'model_dir'


def log_output(i, grad, img):
    # scipy.misc.imsave(pp + FGSM_DIR + '/grad_%04d.png' % i, np.reshape(grad, [28, 28]))
    scipy.misc.imsave(pp + FGSM_DIR + '/img_%04d.png' % i, np.reshape(img, [28, 28]))


def fgsm(model_name):
    """
    Loads the given model and computes an adversarial example using FGSM (fast gradient sign method).
    Instead of computing the gradient of the loss we use the element of the probability output vector that corresponds
    to the sample class.
    """
    if not os.path.exists(FGSM_DIR):
        os.makedirs(FGSM_DIR)

    tf.reset_default_graph()

    # computational graph
    img_batch = tf.placeholder(tf.float32, shape=[None, 28 * 28], name='img_batch')
    out = model.cnn(img_batch)
    probabilities = out.get('probabilities')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, pp + MODEL_DIR + os.sep + 'model_' + model_name + '.ckpt')
        mnist_class = 0
        a = data.get_class_sample(mnist_class)

        # compute partial y / partial x
        # how does the specific class prediction depend on every single input
        gradients = tf.gradients(probabilities[:, mnist_class], img_batch)

        grad_vals = sess.run(gradients, feed_dict={img_batch: [a]})[0]  # unpack batch
        grad_vals = grad_vals[0]  # first class (selection above)
        grad_vals_sign = np.sign(grad_vals) * 1.0/255.  # could be used for the sign method
        scipy.misc.imsave(pp + FGSM_DIR + '/grad.png', np.reshape(np.abs(grad_vals), [28, 28]))
        scipy.misc.imsave(pp + FGSM_DIR + '/grad_sign.png', np.reshape(grad_vals_sign, [28, 28]))

        # scaling of the attack vector
        epss = np.arange(0., 50., 1)  # epsilon values
        attacks = [a - grad_vals_sign * 1 * x for x in epss]
        attacks = np.clip(attacks, 0, 1)  # clip image pixels to [0,1]

        # compute probabilities for the attack images
        probabilities_out = sess.run(probabilities, feed_dict={img_batch: attacks})

        for i in range(len(probabilities_out)):
            log_output(i, grad_vals_sign * epss[i], attacks[i])

        # log classes
        scipy.misc.imsave(pp + FGSM_DIR + '/probabilities.png', probabilities_out)
        print probabilities_out
        print np.argmax(probabilities_out, axis=1)


def get_attack_batch(model_name, count):
    if not os.path.exists(FGSM_DIR):
        os.makedirs(FGSM_DIR)

    tf.reset_default_graph()

    # computational graph
    img_batch = tf.placeholder(tf.float32, shape=[None, 28*28], name='img_batch')
    label_batch = tf.placeholder(tf.float32, shape=[None, 10], name='labels_batch')
    out = model.cnn(img_batch)
    logits = out.get('logits')
    probabilities = out.get('probabilities')
    loss = model.loss(label_batch, logits)

    img_batch_val, label_batch_val = data.get_test_batch(count)
    classes_batch_val = np.argmax(label_batch_val, axis=1)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, pp + MODEL_DIR + os.sep + 'model_' + model_name + '.ckpt')

        gradients = tf.gradients(loss, img_batch)
        grad_vals, probabilities_val = sess.run([gradients, probabilities], feed_dict={
            img_batch: img_batch_val,
            label_batch: label_batch_val
        })

        grad_vals_sign = np.sign(grad_vals[0]) * 1.0 / 255.
        assigned_classes = np.argmax(probabilities_val, axis=1)

        original_images = []
        successful_attacks = []

        for i, grad in enumerate(grad_vals_sign):
            if assigned_classes[i] != classes_batch_val[i]:
                # classification should have been correct
                continue
            epss = np.arange(0., 100., 1)  # epsilon values
            attacks = [img_batch_val[i] + grad * 1 * x for x in epss]
            attacks = np.clip(attacks, 0, 1)  # clip image pixels to [0,1]

            # run classification on attacks
            probabilities_val = sess.run(probabilities, feed_dict={img_batch: attacks})

            best_attack = get_first_successful(probabilities_val, attacks)
            if best_attack is not None:
                successful_attacks.append(best_attack)
                original_images.append(img_batch_val[i])

    log_attacks(original_images, successful_attacks)
    return original_images, successful_attacks


def get_first_successful(probabilities, attacks):
    assert len(probabilities) > 0
    classes = np.argmax(probabilities, axis=1)

    for i, assigned_class in enumerate(classes):
        if assigned_class != classes[0]:
            return attacks[i]
    return None


def log_attacks(originals, attacks):
    assert len(originals) == len(attacks)
    for i in range(len(originals)):
        combined_img = np.hstack([np.reshape(originals[i], [28, 28]), np.reshape(attacks[i], [28, 28])])
        scipy.misc.imsave('./' + FGSM_DIR + '/attack_%04d.png' % i, combined_img)
