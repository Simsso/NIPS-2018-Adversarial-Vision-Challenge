import tensorflow as tf
from resnet_base.model.submittable_resnet import SubmittableResNet
from resnet_base.data.tiny_imagenet_pipeline import TinyImageNetPipeline


def main(args):
    sess = tf.Session()
    with sess:
        pipeline = TinyImageNetPipeline(batch_size=128)
        imgs, labels = pipeline.get_iterator().get_next()

        model = SubmittableResNet(imgs, labels)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        model.restore(sess)

        # now save (to get a global checkpoint)
        model.save(sess)


if __name__ == "__main__":
    tf.app.run()
