import resnet_base.data.tiny_imagenet as data
import numpy as np
import os
import tensorflow as tf
import resnet_base.util.file_system as file_system
from resnet_base.model.base_model import BaseModel
from resnet_base.model.resnet import ResNet

VALIDATION_BATCH_SIZE = 1024  # does not affect training results; adjustment based on GPU RAM
TF_LOGS = os.path.join('..', 'tf_logs')

CHECKPOINT_DIR = os.path.expanduser('~/.models/tiny_imagenet_alp05_2018_06_26.ckpt')

tf.logging.set_verbosity(tf.logging.DEBUG)


def run_validation(model: BaseModel):
    # data set
    valid_batch = data.batch_q('val', VALIDATION_BATCH_SIZE)

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    summary_merged = tf.summary.merge_all()

    file_system.create_dir(TF_LOGS)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamic GPU memory allocation
    sess = tf.Session(config=config)

    with sess.as_default():
        valid_log_writer = tf.summary.FileWriter(os.path.join(TF_LOGS, 'resnet_valid'), sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run(init)
        model.load(sess)

        try:
            while not coord.should_stop():
                tf.logging.info("Starting evaluation")
                vals = []
                for _ in range(min(data.NUM_VALIDATION_SAMPLES // VALIDATION_BATCH_SIZE, data.NUM_VALIDATION_SAMPLES)):
                    valid_images, valid_labels = sess.run(valid_batch)
                    vals.append(sess.run([model.accuracy, model.loss],
                                         feed_dict={model.x: valid_images, model.labels: valid_labels}))
                    acc_mean_val, loss_mean_val = np.mean(vals, axis=0)
                    print(acc_mean_val)

                tf.logging.info("Validation data: accuracy {}, loss {}".format(acc_mean_val, loss_mean_val))
                break
        except tf.errors.OutOfRangeError as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)


def main(args):
    model = ResNet()
    run_validation(model)


if __name__ == "__main__":
    tf.app.run()
