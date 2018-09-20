import tensorflow as tf
from resnet_base.model.resnet import ResNet
from resnet_base.trainer.resnet_trainer import ResNetTrainer
from resnet_base.data.tiny_imagenet_pipeline import TinyImageNetPipeline


def main(args):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamic GPU memory allocation
    sess = tf.Session(config=config)

    pipeline = TinyImageNetPipeline()
    imgs, labels = pipeline.get_iterator().get_next()
    model = ResNet(imgs, labels)

    trainer = ResNetTrainer(sess, model, pipeline)
    trainer.train()


if __name__ == "__main__":
    tf.app.run()
