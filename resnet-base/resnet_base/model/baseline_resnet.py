from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from resnet_base.model.base_model import BaseModel
from resnet_base.data.tiny_imagenet_pipeline import TinyImageNetPipeline as Data, TinyImageNetPipeline
from resnet_base.util.logger.factory import LoggerFactory


# define flags
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("baseline_checkpoint", "", "Path to the baseline weights checkpoint (used to restore only")


class BaselineResNet(BaseModel):

    def __init__(self, logger_factory: LoggerFactory = None, x: tf.Tensor = None, labels: tf.Tensor = None):
        super().__init__(logger_factory)

        self.accuracy = None  # percentage of correctly classified samples
        self.loss = None
        self.logits = None
        self.softmax = None

        if x is None:
            x = tf.placeholder(tf.float32, name='x', shape=[None, Data.img_width, Data.img_height, Data.img_channels])
        self.x = x

        if labels is None:
            labels = tf.placeholder(tf.uint8, shape=[None], name='labels')
        self.labels = labels

        self.is_training = tf.placeholder_with_default(False, (), 'is_training')
        self.num_classes = TinyImageNetPipeline.num_classes

        with tf.variable_scope('custom') as scope:
            self.custom_scope = scope

        self.logits = self._build_model(self.x)
        self.softmax = tf.nn.softmax(self.logits, name='predictions')

        self.baseline_saver: tf.train.Saver = None

        self._init_loss()
        self._init_accuracy()

        self.post_build_init()

    def _build_model(self, raw_imgs: tf.Tensor) -> tf.Tensor:
        """
        Builds the ResNet model graph with the TF API. This function is intentionally kept simple and sequential to
        simplify the addition of new layers.
        :param raw_imgs: Input to the model, i.e. an image batch
        :return: Logits of the model
        """
        processed_imgs = BaselineResNet.__baseline_preprocessing(raw_imgs)
        first_conv = BaselineResNet.__conv2d_fixed_padding(inputs=processed_imgs, filters=64, kernel_size=3, strides=1)

        # blocks
        block1 = BaselineResNet.__block_layer(first_conv, filters=64, strides=1, is_training=self.is_training, index=1)
        block2 = BaselineResNet.__block_layer(block1, filters=128, strides=2, is_training=self.is_training, index=2)
        block3 = BaselineResNet.__block_layer(block2, filters=256, strides=2, is_training=self.is_training, index=3)
        block4 = BaselineResNet.__block_layer(block3, filters=512, strides=2, is_training=self.is_training, index=4)

        block4_norm = BaselineResNet.__batch_norm(block4, self.is_training)
        block4_postact = tf.nn.relu(block4_norm)

        global_avg = tf.reduce_mean(block4_postact, [1, 2], keepdims=True)
        global_avg = tf.identity(global_avg, 'final_reduce_mean')

        global_avg = tf.reshape(global_avg, [-1, 512])
        dense = tf.layers.Dense(units=self.num_classes, name='readout_layer')(global_avg)
        dense = tf.identity(dense, 'final_dense')

        self.activations = {
            'act0_raw_input': raw_imgs,
            'act1_processed_imgs': processed_imgs,
            'act2_first_conv': first_conv,
            'act3_block1': block1,
            'act4_block2': block2,
            'act5_block3': block3,
            'act6_block4': block4,
            'act7_block4_postact': block4_postact,
            'act8_global_avg': global_avg,
            'act9_logits': dense
        }

        return dense

    def init_saver(self) -> None:
        """
        Creates two savers:
            - baseline_saver for all baseline weights (restore-only)
            - saver for all variables in the graph (save-only)
        """
        all_vars = set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, ''))
        custom_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.custom_scope.name)
        meta_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'meta')
        for var in custom_vars + meta_vars:
            all_vars.remove(var)
        baseline_var_list = list(all_vars)

        self.baseline_saver = BaseModel._create_saver_from_var_list(baseline_var_list)
        self.saver = BaseModel._create_saver('')

    def save(self, sess: tf.Session) -> None:
        # only save the full graph to the save_dir
        BaseModel._save_to_path(sess, self.saver, self.global_step, path=FLAGS.save_dir)

    def restore(self, sess: tf.Session) -> None:
        if FLAGS.baseline_checkpoint:
            # only restore the baseline checkpoint
            BaseModel._restore_checkpoint(self.baseline_saver, sess, path=FLAGS.baseline_checkpoint)
        else:
            # (try to) restore the full graph
            BaseModel._restore_checkpoint(self.saver, sess, path=FLAGS.save_dir)

    def _init_loss(self) -> None:
        """
        Adds a classification cross entropy loss term to the LOSSES collection.
        Initializes the loss attribute from the LOSSES collection (sum of all entries).
        """
        labels_one_hot = tf.one_hot(self.labels, depth=Data.num_classes)
        cross_entropy_loss = tf.losses.softmax_cross_entropy(labels_one_hot, self.logits)
        # cross_entropy_loss is a scalar
        tf.add_to_collection(tf.GraphKeys.LOSSES, cross_entropy_loss)
        self.loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES))
        self.logger_factory.add_scalar('loss', self.loss, log_frequency=10)
        self.logger_factory.add_scalar('cross_entropy_loss', cross_entropy_loss, log_frequency=25)

    def _init_accuracy(self) -> None:
        """
        Initializes the accuracy attribute. It is the percentage of correctly classified samples (value in [0,1]).
        """
        correct = tf.cast(tf.equal(tf.argmax(self.softmax, axis=1), tf.cast(self.labels, tf.int64)), dtype=tf.float32)
        self.accuracy = tf.reduce_mean(correct, name='accuracy')
        self.logger_factory.add_scalar('accuracy', self.accuracy, log_frequency=10)

    @staticmethod
    def __baseline_preprocessing(x: tf.Tensor) -> tf.Tensor:
        """
        Performs preprocessing on a standard Tiny ImageNet input which is expected to be in [-1, 1].
        :param x: the input Tensor
        :return: a rescaled input tensor
        """
        # the baseline ResNet expects [0, 255]
        x = (x + tf.constant(1.)) * 127.5

        # preprocessing
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        _CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

        return x - tf.constant(_CHANNEL_MEANS)

    @staticmethod
    def __conv2d_fixed_padding(inputs, filters, kernel_size, strides):
        """Strided 2-D convolution with explicit padding."""
        # The padding is consistent and is based only on `kernel_size`, not on the
        # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
        if strides > 1:
            inputs = BaselineResNet.__fixed_padding(inputs, kernel_size)

        return tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer())

    @staticmethod
    def __block_layer(inputs, filters, strides, is_training, index):
        """Creates one layer of blocks for the ResNet model.

        Args:
          inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
          filters: The number of filters for the first convolution of the layer.
          strides: The stride to use for the first convolution of the layer. If
            greater than 1, this layer will ultimately downsample the input.
          is_training: Either True or False, whether we are currently training the
            model. Needed for batch norm.

        Returns:
          The output tensor of the block layer.
        """
        def projection_shortcut(inputs):
            return BaselineResNet.__conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1, strides=strides)

        # Only the first block per block_layer uses projection_shortcut and strides
        inputs = BaselineResNet._building_block_v2(inputs, filters, is_training, projection_shortcut, strides)
        inputs = BaselineResNet._building_block_v2(inputs, filters, is_training, None, 1)

        return tf.identity(inputs, "block_layer{}".format(index))

    @staticmethod
    def __fixed_padding(inputs, kernel_size):
        """Pads the input along the spatial dimensions independently of input size.

        Args:
          inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
          kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                       Should be a positive integer.

        Returns:
          A tensor with the same format as the input with the data either intact
          (if kernel_size == 1) or padded (if kernel_size > 1).
        """
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return padded_inputs

    @staticmethod
    def _building_block_v2(inputs, filters, training, projection_shortcut, strides):
        """A single block for ResNet v2, without a bottleneck.

        Batch normalization then ReLu then convolution as described by:
          Identity Mappings in Deep Residual Networks
          https://arxiv.org/pdf/1603.05027.pdf
          by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

        Args:
          inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
          filters: The number of filters for the convolutions.
          training: A Boolean for whether the model is in training or inference
            mode. Needed for batch normalization.
          projection_shortcut: The function to use for projection shortcuts
            (typically a 1x1 convolution when downsampling the input).
          strides: The block's stride. If greater than 1, this block will ultimately
            downsample the input.

        Returns:
          The output tensor of the block; shape should match inputs.
        """
        shortcut = inputs
        inputs = BaselineResNet.__batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)

        # The projection shortcut should come after the first batch norm and ReLU
        # since it performs a 1x1 convolution.
        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)

        inputs = BaselineResNet.__conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3,
                                                            strides=strides)

        inputs = BaselineResNet.__batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)
        inputs = BaselineResNet.__conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=1)

        return inputs + shortcut

    @staticmethod
    def __batch_norm(inputs, training):
        """Performs a batch normalization using a standard set of parameters."""
        return tf.layers.batch_normalization(inputs=inputs, axis=3, momentum=0.997, epsilon=1e-5, center=True,
                                             scale=True, training=training, fused=True)
