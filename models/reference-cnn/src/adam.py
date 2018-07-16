import tensorflow as tf
from simple_cnn import simple_cnn
from utils.config import config

save_path = "%s/%s" % (config['checkpoint_save_path'], config['model_name'])


def train(train_images, train_labels, num_steps):
    optimizer = tf.train.AdamOptimizer(config['learning_rate'])

    x = tf.placeholder(tf.float32, shape=[None, config['image_size'], config['image_size'], 3])
    y = tf.placeholder(tf.int32)

    loss = simple_cnn(input_x=x, labels=y, mode=tf.estimator.ModeKeys.TRAIN)
    train_op = optimizer.minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # saver.restore(sess, save_path)
        for i in range(num_steps):
            batch_x = train_images[(i * config['batch_size']):((i + 1) * config['batch_size']), :, :, :]
            batch_y = train_labels[(i * config['batch_size']):((i + 1) * config['batch_size'])]

            _, loss_value = sess.run([train_op, loss], feed_dict={
                x: batch_x,
                y: batch_y
            })

            print("Loss in iteration %d: %f" % (i, loss_value))

        saver.save(sess, save_path)
