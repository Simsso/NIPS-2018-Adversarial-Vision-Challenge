import model
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

LEARNING_RATE = 3e-4
STEPS = 10000
BATCH_SIZE = 100
MODEL_NAME = '2'

# mnist
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)


def train():
    # computational graph
    img_batch = tf.placeholder(tf.float32, shape=[None, 28*28], name='img_batch')
    labels_batch = tf.placeholder(tf.float32, shape=[None, 10], name='labels_batch')
    out = model.cnn(img_batch)
    logits = out.get('logits')
    probabilities = out.get('probabilities')
    loss = model.loss(labels_batch, logits)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # tensor board logging
    tf.summary.image('input', tf.reshape(img_batch, [-1, 28, 28, 1]), max_outputs=4)
    summary_merged = tf.summary.merge_all()

    # init
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)
    log_writer = tf.summary.FileWriter('tf_logs' + os.sep + MODEL_NAME, sess.graph)

    for step in range(STEPS):
        img_batch_val, labels_batch_val = mnist.train.next_batch(BATCH_SIZE)
        _, summary = sess.run([optimizer, summary_merged], feed_dict={
            img_batch: img_batch_val,
            labels_batch: labels_batch_val
        })

        if step % 10 == 0:
            log_writer.add_summary(summary, step)
