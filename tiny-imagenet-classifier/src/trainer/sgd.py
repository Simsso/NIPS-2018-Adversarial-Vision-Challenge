import data.tiny_imagenet as data
import numpy as np
import os
import tensorflow as tf
import util.file_system

LEARNING_RATE = .002
NUM_EPOCHS = 1000
TRAIN_BATCH_SIZE = 64
VALIDATION_BATCH_SIZE = 64  # does not affect training results; adjustment based on GPU RAM
STEPS_PER_EPOCH = min(data.NUM_TRAIN_SAMPLES // TRAIN_BATCH_SIZE, data.NUM_TRAIN_SAMPLES)
TF_LOGS = os.path.join('..', 'tf_logs')
WEIGHT_DECAY = 1e-4
DROPOUT = .5

def saver_path(model_name):
    return "checkpoints/" + model_name + "_model.ckpt"

def train(model_def):
    def run_validation():
        vals = []
        for _ in range(min(data.NUM_VALIDATION_SAMPLES // VALIDATION_BATCH_SIZE, data.NUM_VALIDATION_SAMPLES)):
            valid_images, valid_labels = sess.run(valid_batch)
            vals.append(sess.run([acc, loss], feed_dict={images: valid_images, labels: valid_labels}))
        acc_mean_val, loss_mean_val = np.mean(vals, axis=0)
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='accuracy', simple_value=acc_mean_val),
            tf.Summary.Value(tag='loss', simple_value=loss_mean_val),
        ])
        valid_log_writer.add_summary(summary, global_step=(epoch - 1) * STEPS_PER_EPOCH)

        # log the non-artificial summary as well
        valid_images, valid_labels = sess.run(valid_batch)
        summary = sess.run(summary_merged, feed_dict={images: valid_images, labels: valid_labels})
        valid_log_writer.add_summary(summary, global_step=(epoch - 1) * STEPS_PER_EPOCH)

        return acc_mean_val, loss_mean_val

    def run_training():
        vals = []
        for _ in range(STEPS_PER_EPOCH):
            train_images, train_labels = sess.run(train_batch)
            _, acc_val, loss_val = sess.run([optimizer, acc, loss],
                                            feed_dict={images: train_images, labels: train_labels, is_training: True})
            vals.append([acc_val, loss_val])
        acc_mean_val, loss_mean_val = np.mean(vals, axis=0)
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='accuracy', simple_value=acc_mean_val),
            tf.Summary.Value(tag='loss', simple_value=loss_mean_val),
        ])
        train_log_writer.add_summary(summary, global_step=epoch * STEPS_PER_EPOCH)

        return acc_mean_val, loss_mean_val

    graph = tf.Graph()
    with graph.as_default():
        # inputs (images and labels)
        images = tf.placeholder(tf.float32, shape=[None, data.IMG_DIM, data.IMG_DIM, data.IMG_CHANNELS], name='images')
        labels = tf.placeholder(tf.uint8, shape=[None], name='labels')
        is_training = tf.placeholder_with_default(False, (), 'is_training')

        # data set
        train_batch = data.batch_q('train', TRAIN_BATCH_SIZE)
        valid_batch = data.batch_q('val', VALIDATION_BATCH_SIZE)

        logits, softmax = model_def.graph(tf.cast(images, tf.float32), is_training, DROPOUT, WEIGHT_DECAY)
        loss = model_def.loss(labels, logits)
        acc = model_def.accuracy(labels, softmax)
        optimizer = get_optimization_op(loss)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        summary_merged = tf.summary.merge_all()
        
        saver = tf.train.Saver()

    util.file_system.create_dir(TF_LOGS)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamic GPU memory allocation
    sess = tf.Session(graph=graph, config=config)

    previous_val_acc = -1
    number_of_epochs_decreased = 0

    with sess.as_default():
        train_log_writer = tf.summary.FileWriter(os.path.join(TF_LOGS, '%s_train' % model_def.NAME), sess.graph)
        valid_log_writer = tf.summary.FileWriter(os.path.join(TF_LOGS, '%s_valid' % model_def.NAME), sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run(init)
        # saver.restore(sess, saver_path(model_def.NAME))

        try:
            while not coord.should_stop():
                for i in range(NUM_EPOCHS):
                    epoch = i + 1
                    print("Starting epoch #{}".format(epoch))
                    valid_acc, valid_loss = run_validation()
                    print("Validation data: accuracy {}, loss {}".format(valid_acc, valid_loss))

                    if valid_acc > previous_val_acc:
                        number_of_epochs_decreased = 0
                        previous_val_acc = valid_acc
                        if valid_acc > 0.2:
                            saver.save(sess, saver_path(model_def.NAME))
                            print("Saved current weights to %s" % (saver_path(model_def.NAME)))
                    elif number_of_epochs_decreased > 5:
                        print("Too many epochs with decreasing validation accuracy. Stopping training.")
                        break
                    else:
                        number_of_epochs_decreased += 1

                    valid_acc, valid_loss = run_training()
                    print("Training data (with batch norm): accuracy {}, loss {}".format(valid_acc, valid_loss))
                break
        except tf.errors.OutOfRangeError as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)


def get_optimization_op(loss):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        return optimizer.minimize(loss)
