from glob import glob
from os import path
import re
import tensorflow as tf
from typing import Dict, List, Optional, Tuple

from resnet_base.data.base_pipeline import BasePipeline


tf.flags.DEFINE_string('data_dir', path.expanduser('~/.data/tiny-imagenet-200'),
                       'Path of the Tiny ImageNet dataset folder')

FLAGS = tf.flags.FLAGS


class TinyImageNetPipeline(BasePipeline):
    """
    Input pipeline for the Tiny ImageNet dataset (https://tiny-imagenet.herokuapp.com/). Streams for training and
    validation samples.
    """
    num_classes = 200
    num_train_samples = 500 * num_classes
    num_valid_samples = 50 * num_classes
    img_width = 64
    img_height = img_width
    img_channels = 3

    __supported_modes = [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]

    def __init__(self, data_dir: str = None, batch_size: int = 256):
        """
        :param data_dir: Directory of the folder from http://cs231n.stanford.edu/tiny-imagenet-200.zip
        :param batch_size: Batch size used for training and validation.
        """
        super().__init__()
        if not data_dir:
            data_dir = FLAGS.data_dir
        self.data_dir = data_dir

        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0. Got '{}'.".format(batch_size))
        self.batch_size = batch_size

        self.placeholder = {
            tf.estimator.ModeKeys.TRAIN: (
                tf.placeholder(tf.string, self.num_train_samples),
                tf.placeholder(tf.string, self.num_train_samples)
            ),
            tf.estimator.ModeKeys.EVAL: (
                tf.placeholder(tf.string, self.num_valid_samples),
                tf.placeholder(tf.string, self.num_valid_samples)
            )
        }
        self.filenames = {}
        self.raw_labels = {}

        self.__init_filenames_labels()

        # init_ops must be created prior to session instantiation
        self._get_init_op(tf.estimator.ModeKeys.TRAIN)
        self._get_init_op(tf.estimator.ModeKeys.EVAL)

    def _construct_iterator(self) -> tf.data.Iterator:
        """
        Creates the iterator from structures: The types are float32 (image data) and uint8 (label index). The output
        shapes are ((batch_img, img_width, img_height, channels), batch_labels).
        :return: Iterator for batches of Tiny ImageNet samples
        """
        output_types = (tf.float32, tf.uint8)
        output_shapes = tf.TensorShape((None, self.img_width, self.img_height, self.img_channels)), tf.TensorShape(None)
        return tf.data.Iterator.from_structure(output_types, output_shapes)

    def _construct_init_op(self, mode: tf.estimator.ModeKeys) -> tf.Operation:
        """
        Constructs the actual tf.data.Dataset used in the input pipeline. Applies several mapping functions. Image
        augmentation is only used if the mode is TRAIN.
        :param mode: TRAIN (training) or EVAL (validation)
        :return: Initializer operation which can be used to switch to the given mode.
        """
        if mode not in self.__supported_modes:
            raise ValueError("Supported modes are {}. Got '{}'.".format(self.__supported_modes, mode))

        data = tf.data.Dataset.from_tensor_slices(self.placeholder[mode])
        data = data.shuffle(buffer_size=self.__get_num_samples(mode))
        data = data.map(self.__img_loading)
        if mode == tf.estimator.ModeKeys.TRAIN:
            data = data.map(self.__img_augmentation)
        data = data.map(self.__img_label_scaling)
        data = data.batch(self.batch_size)
        iterator = self.get_iterator()
        return iterator.make_initializer(data)

    def __img_loading(self, img_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Loads an image from the given path (TF graph).
        :param img_path: Path pointing to the image
        :param label: Label of the image (not used because the function is used with dataset map)
        :return: Tuple of (img, label), where img is a tensor with float32 values in [0,1]
        """
        file = tf.read_file(img_path)
        img = tf.image.decode_jpeg(file, self.img_channels)  # uint8 [0, 255]
        img = tf.cast(img, tf.float32) / 255  # float32 [0., 1.]
        return img, label

    def __img_augmentation(self, img: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Augments an image tensor.
        :param img: Image to augment, values in [0,1]
        :param label: Label of the image (not used because the function is used with dataset map)
        :return: Tuple of (img, label), where img is a randomly augmented version of the input image.
        """
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_hue(img, 0.05)
        img = tf.image.random_saturation(img, 0.5, 1.5)
        return img, label

    def __img_label_scaling(self, img: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Scaling casting and set_shape calls which are needed in the chain of dataset map function calls.
        :param img: Image with values in [0,1]
        :param label: Label of the image
        :return: Tuple of (img, label), where img is in [-1,1] and label is a uint8 (Tiny ImageNet has 200 classes)
        """
        img = tf.multiply(2., img) - 1.  # float32 [-1., 1]
        img.set_shape([self.img_width, self.img_height, self.img_channels])

        label = tf.string_to_number(label, tf.int32)
        label = tf.cast(label, tf.uint8)
        return img, label

    def __init_filenames_labels(self) -> None:
        """
        Initializes the filenames and raw_labels attributes which are held in RAM.
        """
        self.filenames = {}
        self.raw_labels = {}
        for mode in self.__supported_modes:
            self.filenames[mode], self.raw_labels[mode] = self.__load_filenames_labels(mode)

    def __load_filenames_labels(self, mode: tf.estimator.ModeKeys) -> Tuple[List[str], List[str]]:
        """
        Reads the dataset folder filenames and labels of the given split (training data or validation data).
        :param mode: TRAIN (training) or EVAL (validation)
        :return: List of filenames and a associated labels (two lists of equal length).
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
        """
        :return: Tuple with two dictionaries:
                 Dictionary from class name to class index (e.g. 'n01944390' --> 4)
                 Dictionary from class index to class description (e.g. 6 --> 'dog')
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
        """
        :param mode: TRAIN (training) or EVAL (validation)
        :return: Number of samples the dataset contains for the given pipeline mode.
        """
        return self.num_train_samples if mode == tf.estimator.ModeKeys.TRAIN else self.num_valid_samples

    def switch_to(self, mode: tf.estimator.ModeKeys, feed_dict: Optional[Dict] = None, sess: tf.Session = None) -> None:
        """
        Switches the input pipeline to the given mode in the given session.
        :param mode: TRAIN (training) or EVAL (validation)
        :param feed_dict: Feed dictionary that will be passed when evaluating the initialization op. Defaults to the
                          dataset data.
        :param sess: Session to switch the mode in. Defaults to the tf.get_default_session() value.
        """
        if feed_dict is None:
            feed_dict = {
                self.placeholder[mode][0]: self.filenames[mode],
                self.placeholder[mode][1]: self.raw_labels[mode]
            }
        super(TinyImageNetPipeline, self).switch_to(mode, feed_dict)
