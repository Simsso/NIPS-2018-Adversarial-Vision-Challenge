import data
import model
import os
import tensorflow as tf

LEARNING_RATE = 3e-4
STEPS = 200
BATCH_SIZE = 100
MODEL_NAME = '4'


def train():
    # computational graph
    img_batch = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 28*28], name='img_batch')
    labels_batch = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 10], name='labels_batch')
    out = model.cnn(img_batch)
    logits = out.get('logits')
    probabilities = out.get('probabilities')
    loss = model.loss(labels_batch, logits)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # tensor board logging
    tf.summary.image('input', tf.reshape(img_batch, [-1, 28, 28, 1]), max_outputs=4)
    summary_merged = tf.summary.merge_all()

    # add ops to save and restore all the variables
    saver = tf.train.Saver()

    # init
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    log_writer = tf.summary.FileWriter('tf_logs' + os.sep + MODEL_NAME, sess.graph)

    for step in range(STEPS):
        print(step)
        img_batch_val, labels_batch_val = data.mnist.train.next_batch(BATCH_SIZE)
        _, summary, _ = sess.run([optimizer, summary_merged, probabilities], feed_dict={
            img_batch: img_batch_val,
            labels_batch: labels_batch_val
        })

        if step % 10 == 0:
            log_writer.add_summary(summary, step)

    print("completed gradient descend")

    save_path = saver.save(sess, "model_dir/model_" + MODEL_NAME + ".ckpt")
    print("model saved at %s" % save_path)
