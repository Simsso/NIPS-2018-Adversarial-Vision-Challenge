import tensorflow as tf
from nips_defense.model.resnet import ResNet
from nips_defense.trainer.resnet_trainer import ResNetTrainer
from nips_defense.data.tiny_imagenet_pipeline import TinyImageNetPipeline


def main(args):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamic GPU memory allocation
    sess = tf.Session(config=config)

    with sess.as_default():
        pipeline = TinyImageNetPipeline()
        imgs, labels = pipeline.get_iterator().get_next()
        model = ResNet(imgs, labels)

        trainer = ResNetTrainer(model, pipeline)
        trainer.train()


if __name__ == "__main__":
    tf.app.run()
