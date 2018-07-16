from utils.tiny_image_net import TinyImageNet
from adam import train
import tensorflow as tf


def main(unused_argv):
    tiny_image_net = TinyImageNet()
    train_images, train_labels = tiny_image_net.get_train_data()
    train(train_images, train_labels, num_steps=10)
    print("Done training. TODO: evaluation metrics")


if __name__ == "__main__":
    print("Launching Reference-CNN Test...")
    tf.app.run()
