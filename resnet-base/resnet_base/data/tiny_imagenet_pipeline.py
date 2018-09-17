from glob import glob
from os import path
import re
import tensorflow as tf
from typing import Dict, List, Tuple

from resnet_base.data.base_pipeline import BasePipeline


tf.flags.DEFINE_string('data_dir', path.expanduser('~/.data/tiny-imagenet-200'),
                       'Path of the Tiny ImageNet dataset folder')

FLAGS = tf.flags.FLAGS


class TinyImageNetPipeline(BasePipeline):
    num_classes = 200
    num_train_samples = 500 * num_classes
    num_valid_samples = 50 * num_classes
    img_width = 64
    img_height = img_width
    img_channels = 3

    __supported_modes = [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]

    def __init__(self, data_dir: str = None, batch_size: int = 256):
        super().__init__()
        if not data_dir:
            data_dir = FLAGS.data_dir
        self.data_dir = data_dir

        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0. Got '{}'.".format(batch_size))
        self.batch_size = batch_size

    def _construct_iterator(self) -> tf.data.Iterator:
        output_types = (tf.float32, tf.uint8)
        output_shapes = tf.TensorShape((None, self.img_width, self.img_height, self.img_channels)), tf.TensorShape(None)
        return tf.data.Iterator.from_structure(output_types, output_shapes)

    def _construct_init_op(self, mode: tf.estimator.ModeKeys) -> tf.Operation:
        if mode not in self.__supported_modes:
            raise ValueError("Supported modes are {}. Got '{}'.".format(self.__supported_modes, mode))

        filenames, labels = self.__load_filenames_labels(mode)  # TODO: switch to non-tf.Constant way of storing here
        data = tf.data.Dataset.from_tensor_slices((filenames, labels))
        data = data.shuffle(buffer_size=self.__get_num_samples(mode))
        data = data.map(self.__img_loading)
        if mode == tf.estimator.ModeKeys.TRAIN:
            data = data.map(self.__img_augmentation)
        data = data.map(self.__img_label_scaling)
        data = data.batch(self.batch_size)
        iterator = self.get_iterator()
        return iterator.make_initializer(data)

    def __img_loading(self, img_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        file = tf.read_file(img_path)
        img = tf.image.decode_jpeg(file, self.img_channels)  # uint8 [0, 255]
        img = tf.cast(img, tf.float32) / 255  # float32 [0., 1.]
        return img, label

    def __img_augmentation(self, img: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_hue(img, 0.05)
        img = tf.image.random_saturation(img, 0.5, 1.5)
        return img, label

    def __img_label_scaling(self, img: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        img = tf.multiply(2., img) - 1.  # float32 [-1., 1]
        img.set_shape([self.img_width, self.img_height, self.img_channels])

        label = tf.string_to_number(label, tf.int32)
        label = tf.cast(label, tf.uint8)
        return img, label

    def __load_filenames_labels(self, mode: tf.estimator.ModeKeys) -> Tuple[List[str], List[str]]:
        """Reads the dataset folder and returns a list of filenames and a associated labels (therefore equal length).
        """
        label_dict, class_description = self.__build_label_dicts()
        filenames, labels = [], []
        if mode == tf.estimator.ModeKeys.TRAIN:
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

    def __build_label_dicts(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Returns a dictionary from class name to class index (e.g. 'n01944390' --> 4) and a dictionary from
        class index to class description (e.g. 6 --> 'dog')
        """
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

    def __get_num_samples(self, mode: tf.estimator.ModeKeys) -> int:
        """Returns the number of samples which the dataset contains for the chosen training mode.
        """
        return self.num_train_samples if mode == tf.estimator.ModeKeys.TRAIN else self.num_valid_samples
