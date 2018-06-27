import tensorflow as tf
from ..model.simple_cnn import simple_cnn
from ..data.tiny_image_net import IMAGE_SIZE

LEARNING_RATE = 0.01
BATCH_SIZE = 50
NUM_STEPS = 10


def train(train_images, train_labels):
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
    y = tf.placeholder(tf.int32)

    with tf.Session() as session:
        for i in range(NUM_STEPS):
            batch_x = train_images[(i * BATCH_SIZE):((i+1)*BATCH_SIZE), :, :, :]
            batch_y = train_labels[(i * BATCH_SIZE):((i+1)*BATCH_SIZE), :, :, :]

            loss = simple_cnn(x, y, mode=tf.estimator.ModeKeys.TRAIN)
            session.run(optimizer.minimize(loss), feed_dict={
                x: batch_x,
                y: batch_y
            })

            loss_value = session.run(loss, feed_dict={
                x: batch_x,
                y: batch_y
            })

            print("Loss in iteration %d: %f" % (i, loss_value))

