import tensorflow as tf
from nips_defense.model.baseline_lesci_resnet import BaselineLESCIResNet
from nips_defense.data.tiny_imagenet_pipeline import TinyImageNetPipeline

tf.logging.set_verbosity(tf.logging.DEBUG)


def main(args):
    """
    Saves a model as a checkpoint. Can be used to merge .mat-files and baseline weights into a single checkpoint.
    Sample arguments for a BaselineLESCIResNet:
        --save_dir="/home/timodenk/.models/baseline-lesci"
        --baseline_checkpoint="/home/timodenk/.models/baseline/model.ckpt-5865"
        --lesci_emb_space_file "/home/timodenk/.data/activations/lesci_act5_block3_74246x64.mat"
        --pca_compression_file "/home/timodenk/.data/activations/pca_act5_block3_64.mat"
    In this case, ALP weights and LESCI files would be stored in a single checkpoint.
    """
    imgs, labels = TinyImageNetPipeline().get_iterator().get_next()
    model = BaselineLESCIResNet(x=imgs, labels=labels)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session().as_default() as sess:
        sess.run(init, feed_dict=model.init_feed_dict)
        model.restore(sess)
        model.save(sess)
        tf.logging.info("Successfully saved checkpoint.")


if __name__ == "__main__":
    tf.app.run()
