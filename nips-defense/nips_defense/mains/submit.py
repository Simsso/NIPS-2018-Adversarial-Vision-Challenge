import tensorflow as tf
from adversarial_vision_challenge import model_server

from nips_defense.data.tiny_imagenet_pipeline import TinyImageNetPipeline
from nips_defense.model.submittable_resnet import SubmittableResNet


def main(args):
    # create an input placeholder used by the challenge
    w = TinyImageNetPipeline.img_width
    h = TinyImageNetPipeline.img_height
    c = TinyImageNetPipeline.img_channels

    sess = tf.Session()
    with sess.as_default():
        images = tf.placeholder(tf.float32, (None, w, h, c), name="images")
        model = SubmittableResNet(x=images)
        foolbox_model = model.get_foolbox_model()

    model_server(foolbox_model)


if __name__ == "__main__":
    tf.app.run()
