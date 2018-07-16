import data.tiny_imagenet as data
import model.deep_cnn as cnn
import os
import tensorflow as tf
import util.file_system

LEARNING_RATE = 1e-4
TRAINING_RUN_NAME = 'cnn_002a'
NUM_EPOCHS = 10
TRAIN_BATCH_SIZE = 250
VALID_BATCH_SIZE = 1000
STEPS_PER_EPOCH = int(data.NUM_TRAIN_SAMPLES / TRAIN_BATCH_SIZE)
TF_LOGS = os.path.join('..', 'tf_logs')


def train():
    graph = tf.Graph()
    with graph.as_default():
        # inputs (images and labels)
        images = tf.placeholder(tf.float32, shape=[None, data.IMG_DIM, data.IMG_DIM, data.IMG_CHANNELS], name='images')
        labels = tf.placeholder(tf.uint8, shape=[None], name='labels')

        # data set
        train_batch = data.batch_q('train', TRAIN_BATCH_SIZE, NUM_EPOCHS)
        valid_batch = data.batch_q('val', VALID_BATCH_SIZE, NUM_EPOCHS)

        logits, softmax = cnn.model(tf.cast(images, tf.float32))
        loss = cnn.loss(labels, logits)
        tf.summary.scalar('loss', loss)
        acc = cnn.accuracy(labels, softmax)
        tf.summary.scalar('accuracy', acc)
        optimizer = get_optimization_op(loss)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        summary_merged = tf.summary.merge_all()

    util.file_system.create_dir(TF_LOGS)

    sess = tf.Session(graph=graph)
    with sess.as_default():
        train_log_writer = tf.summary.FileWriter(os.path.join(TF_LOGS, '%s_train' % TRAINING_RUN_NAME), sess.graph)
        valid_log_writer = tf.summary.FileWriter(os.path.join(TF_LOGS, '%s_valid' % TRAINING_RUN_NAME), sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run(init)

        try:
            while not coord.should_stop():
                for epoch in range(NUM_EPOCHS):
                    valid_images, valid_labels = sess.run(valid_batch)
                    summary, acc_val = sess.run([summary_merged, acc],
                                                feed_dict={images: valid_images, labels: valid_labels})
                    valid_log_writer.add_summary(summary, global_step=epoch * STEPS_PER_EPOCH)
                    print(acc_val)

                    for step in range(STEPS_PER_EPOCH):
                        train_images, train_labels = sess.run(train_batch)
                        summary, _ = sess.run([summary_merged, optimizer],
                                              feed_dict={images: train_images, labels: train_labels})
                        train_log_writer.add_summary(summary, global_step=epoch * STEPS_PER_EPOCH + step)
                break
        except tf.errors.OutOfRangeError as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)


def get_optimization_op(loss):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
    return optimizer.minimize(loss)
