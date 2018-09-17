from glob import glob
from os import path
import re
import tensorflow as tf
from typing import Tuple

tf.flags.DEFINE_string('data_dir', path.expanduser('~/.data/tiny-imagenet-200'),
                       'Path of the Tiny ImageNet dataset folder')

FLAGS = tf.flags.FLAGS


class TinyImageNetDataset:
    num_classes = 200
    num_train_samples = 500 * num_classes
    num_valid_samples = 50 * num_classes
    raw_img_dim = 64
    num_channels = 3

    def __init__(self, data_dir: str = None, mode: tf.estimator.ModeKeys = tf.estimator.ModeKeys.TRAIN):
        if not data_dir:
            data_dir = FLAGS.data_dir
        self.data_dir = data_dir

        if mode not in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
            raise ValueError("Parameter 'mode' must be either TRAIN or EVAL.")
        self.mode = mode

        self.num_samples = self.num_train_samples if mode == tf.estimator.ModeKeys.TRAIN else self.num_valid_samples

    def build(self, iterator: tf.data.Iterator, batch_size: int = 256) -> tf.Operation:
        filenames, labels = self.__load_filenames_labels()  # switch to non-tf.Constant way of storing here
        data = tf.data.Dataset.from_tensor_slices((filenames, labels))
        data = data.shuffle(buffer_size=self.num_samples)
        data = data.map(self.__load_image).map(self.__img_preprocesssing)
        data = data.batch(batch_size)
        return iterator.make_initializer(data)

    def __load_image(self, img_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        file = tf.read_file(img_path)
        img = tf.image.decode_jpeg(file, self.num_channels)  # uint8 [0, 255]

        return img, label

    def __img_preprocesssing(self, img: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        img = tf.cast(img, tf.float32) / 255  # float32 [0., 1.]
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_hue(img, 0.05)
            img = tf.image.random_saturation(img, 0.5, 1.5)

        img = img * 2. - 1.  # float32 [-1., 1]
        img.set_shape([self.raw_img_dim, self.raw_img_dim, self.num_channels])

        label = tf.string_to_number(label, tf.int32)
        label = tf.cast(label, tf.uint8)

        return img, label

    def __load_filenames_labels(self):
        label_dict, class_description = self.__build_label_dicts()
        filenames, labels = [], []
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            filenames = glob(path.join(self.data_dir, 'train', '*', 'images', '*.JPEG'))
            filenames.sort()
            for filename in filenames:
                match = re.search(r'n\d+', filename)
                labels.append(str(label_dict[match.group()]))
        else:  # EVAL
            with open(path.join(self.data_dir, 'val', 'val_annotations.txt'), 'r') as f:
                lines = f.readlines()
                lines.sort()
                for line in lines:
                    split_line = line.split('\t')
                    filename = path.join(self.data_dir, 'val', 'images', split_line[0])
                    filenames.append(filename)
                    labels.append(str(label_dict[split_line[1]]))

        return filenames, labels

    def __build_label_dicts(self):
        label_dict, class_description = {}, {}
        with open(path.join(self.data_dir, 'wnids.txt'), 'r') as f:
            for i, line in enumerate(f.readlines()):
                synset = line[:-1]  # remove \n
                label_dict[synset] = i
        with open(path.join(self.data_dir, 'words.txt'), 'r') as f:
            for i, line in enumerate(f.readlines()):
                synset, desc = line.split('\t')
                desc = desc[:-1]  # remove \n
                if synset in label_dict:
                    class_description[label_dict[synset]] = desc

        return label_dict, class_description

    @staticmethod
    def get_iterator() -> tf.data.Iterator:
        return tf.data.Iterator.from_structure(output_types=(tf.float32, tf.uint8),
                                               output_shapes=(tf.TensorShape((None, TinyImageNetDataset.raw_img_dim,
                                                                              TinyImageNetDataset.raw_img_dim,
                                                                              TinyImageNetDataset.num_channels)),
                                                              tf.TensorShape(None)))
