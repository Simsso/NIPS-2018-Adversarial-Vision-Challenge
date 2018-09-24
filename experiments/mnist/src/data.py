from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# mnist
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)


def get_class_sample(digit):
    assert (0 <= digit < 10)
    while True:
        img_batch, label_batch = mnist.train.next_batch(100)
        for i in range(len(label_batch)):
            if np.argmax(label_batch[0]) == digit:
                return img_batch[i]


def get_test_batch(count):
    assert (count >= 0)
    return mnist.test.next_batch(count)
