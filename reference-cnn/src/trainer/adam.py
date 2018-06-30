import tensorflow as tf
from model.simple_cnn import simple_cnn
from data.tiny_image_net import IMAGE_SIZE

LEARNING_RATE = 0.005
BATCH_SIZE = 20

save_path = "checkpoints/simple-cnn-model.ckpt"


def train(train_images, train_labels, num_steps):
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
    y = tf.placeholder(tf.int32)

    loss = simple_cnn(input_x=x, labels=y, mode=tf.estimator.ModeKeys.TRAIN)
    train_op = optimizer.minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # saver.restore(sess, save_path)
        for i in range(num_steps):
            batch_x = train_images[(i * BATCH_SIZE):((i+1)*BATCH_SIZE), :, :, :]
            batch_y = train_labels[(i * BATCH_SIZE):((i+1)*BATCH_SIZE)]

            _, loss_value = sess.run([train_op, loss], feed_dict={
                x: batch_x,
                y: batch_y
            })

            print("Loss in iteration %d: %f" % (i, loss_value))

        saver.save(sess, save_path)

