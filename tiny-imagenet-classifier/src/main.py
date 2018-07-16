import tensorflow as tf
import trainer.sgd as sgd_trainer
import data.tiny_imagenet as data


def main(args=None):
    sgd_trainer.train()
    return
    sess = tf.Session()
    with sess.as_default():
        a, b = data.batch_q('train', 100, 20)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        try:
            while not coord.should_stop():
                c = a.eval()
                print(c)
        except tf.errors.OutOfRangeError as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
