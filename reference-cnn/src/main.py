from data.tiny_image_net import get_train_data
from trainer.adam import train
import tensorflow as tf


def main(unused_argv):
    train_images, train_labels = get_train_data()
    train(train_images, train_labels, num_steps=10)
    print("Done training. TODO: evaluation metrics")


if __name__ == "__main__":
    print("Launching Reference-CNN Test...")
    tf.app.run()
