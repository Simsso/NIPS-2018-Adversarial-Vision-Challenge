import tensorflow as tf
from model.simple_cnn import simple_cnn
from data.tiny_image_net import IMAGE_SIZE

LEARNING_RATE = 0.005
BATCH_SIZE = 20
NUM_STEPS = 10

save_path = "checkpoints/simple-cnn-model.ckpt"


def train(train_images, train_labels):
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
    y = tf.placeholder(tf.int32)

    loss = simple_cnn(input_x=x, labels=y, mode=tf.estimator.ModeKeys.TRAIN)
    train_op = optimizer.minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        #saver.restore(session, save_path)
        for i in range(NUM_STEPS):
            batch_x = train_images[(i * BATCH_SIZE):((i+1)*BATCH_SIZE), :, :, :]
            batch_y = train_labels[(i * BATCH_SIZE):((i+1)*BATCH_SIZE)]

            session.run(train_op, feed_dict={
                x: batch_x,
                y: batch_y
            })

            loss_value = session.run(loss, feed_dict={
                x: batch_x,
                y: batch_y
            })

            print("Loss in iteration %d: %f" % (i, loss_value))

        saver.save(session, save_path)

