import data
import model
import os
import tensorflow as tf

BATCH_SIZE = 100


def run_analysis(model_name):
    tf.reset_default_graph()

    # computational graph
    img_batch = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 28*28], name='img_batch')
    labels_batch = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 10], name='labels_batch')
    out = model.cnn(img_batch)
    logits = out.get('logits')
    probabilities = out.get('probabilities')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "../model_dir" + os.sep + "model_" + model_name + ".ckpt")
        zero = data.getClassSample(0)
        one = data.getClassSample(1)
        result = sess.run([probabilities], feed_dict={
            img_batch: [zero, one] + [[0] * 784] * 98
        })
        print(result)
