import data.inception_transfer_values as data
import numpy as np
import os
import tensorflow as tf
import util.file_system

INITIAL_LEARNING_RATE = 6.5e-4
NUM_EPOCHS = 1000
TRAIN_BATCH_SIZE = 64
VALIDATION_BATCH_SIZE = 64  # does not affect training results; adjustment based on GPU RAM
STEPS_PER_EPOCH = min(data.NUM_TRAIN_SAMPLES // TRAIN_BATCH_SIZE, data.NUM_TRAIN_SAMPLES)
TF_LOGS = os.path.join('..', 'tf_logs')
WEIGHT_DECAY = 0.0116
DROPOUT_RATE = 0.2
LR_DECAY_RATE = 0.713


def random_batch(inputs, labels, batch_size):
    idx = np.random.choice(len(labels), size=batch_size, replace=False)
    x_batch = inputs[idx]
    y_batch = labels[idx]
    return x_batch, y_batch


def train(model_def):
    all_activations_train, all_labels_train = data.get_activations_labels(mode='train')
    all_activations_val, all_labels_val = data.get_activations_labels(mode='val')

    def run_validation():
        vals = []
        full_batches = data.NUM_VALIDATION_SAMPLES // VALIDATION_BATCH_SIZE
        num_batches = full_batches if data.NUM_VALIDATION_SAMPLES % VALIDATION_BATCH_SIZE == 0 else full_batches + 1
        
        for j in range(num_batches):
            from_idx = j * VALIDATION_BATCH_SIZE
            to_idx = min((j + 1) * VALIDATION_BATCH_SIZE, data.NUM_VALIDATION_SAMPLES)
            val_features = all_activations_val[from_idx:to_idx]
            val_labels = all_labels_val[from_idx:to_idx]

            vals.append(sess.run([acc, loss], feed_dict={
                features: val_features,
                labels: val_labels,
                is_training: False
            }))

        acc_mean_val, loss_mean_val = np.mean(vals, axis=0)
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='accuracy', simple_value=acc_mean_val),
            tf.Summary.Value(tag='loss', simple_value=loss_mean_val),
        ])
        valid_log_writer.add_summary(summary, global_step=(epoch - 1) * STEPS_PER_EPOCH)

        # log the non-artificial summary as well (@Simsso don't really know what you meant by this comment)
        random_batch_x, random_batch_y = random_batch(inputs=all_activations_val, labels=all_labels_val, batch_size=VALIDATION_BATCH_SIZE)
        summary = sess.run(summary_merged, feed_dict={
            features: random_batch_x,
            labels: random_batch_y,
            is_training: False
        })
        valid_log_writer.add_summary(summary, global_step=(epoch - 1) * STEPS_PER_EPOCH)

        return acc_mean_val, loss_mean_val

    def run_training():
        vals = []

        full_batches = data.NUM_TRAIN_SAMPLES // TRAIN_BATCH_SIZE
        num_batches = full_batches if data.NUM_TRAIN_SAMPLES % TRAIN_BATCH_SIZE == 0 else full_batches + 1

        for j in range(num_batches):
            from_idx = j * TRAIN_BATCH_SIZE
            to_idx = min((j + 1) * TRAIN_BATCH_SIZE, data.NUM_TRAIN_SAMPLES)
            train_features = all_activations_train[from_idx:to_idx]
            train_labels = all_labels_train[from_idx:to_idx]

            _, acc_val, loss_val = sess.run([optimizer, acc, loss], feed_dict={
                features: train_features,
                labels: train_labels,
                is_training: True
            })
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
        # inputs (features and labels)
        features = tf.placeholder(tf.float32, shape=[None, data.ACTIVATION_DIM], name='features')
        labels = tf.placeholder(tf.uint8, shape=[None], name='labels')
        is_training = tf.placeholder_with_default(False, (), 'is_training')
        
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps=STEPS_PER_EPOCH, decay_rate=LR_DECAY_RATE)

        logits, softmax = model_def.graph(features, is_training, WEIGHT_DECAY, DROPOUT_RATE)

        loss = model_def.loss(labels, logits)
        acc = model_def.accuracy(labels, softmax)
        optimizer = get_optimization_op(loss, learning_rate, global_step) 

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        summary_merged = tf.summary.merge_all()

    util.file_system.create_dir(TF_LOGS)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamic GPU memory allocation
    sess = tf.Session(graph=graph, config=config)

    with sess.as_default():
        train_log_writer = tf.summary.FileWriter(os.path.join(TF_LOGS, '%s_train' % model_def.NAME), sess.graph)
        valid_log_writer = tf.summary.FileWriter(os.path.join(TF_LOGS, '%s_valid' % model_def.NAME), sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run(init)

        try:
            while not coord.should_stop():
                for i in range(NUM_EPOCHS):
                    epoch = i + 1
                    print("Starting epoch #{}".format(epoch))

                    valid_acc, valid_loss = run_validation()
                    print("Validation data: accuracy {}, loss {}".format(valid_acc, valid_loss))

                    train_acc, train_loss = run_training()
                    print("Training data: accuracy {}, loss {}".format(train_acc, train_loss))
                break
        except tf.errors.OutOfRangeError as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)


def get_optimization_op(loss, learning_rate, global_step):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return optimizer.minimize(loss, global_step=global_step)    # passing the global step here will increase it at every step
