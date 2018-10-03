import os
from resnet_base.model.base_model import BaseModel
from resnet_base.data.tiny_imagenet_pipeline import TinyImageNetPipeline as Data, TinyImageNetPipeline
import tensorflow as tf
from typing import Dict, Optional

from resnet_base.util.logger.factory import LoggerFactory

slim = tf.contrib.slim

# define flags
tf.flags.DEFINE_string("pretrained_checkpoint", "", "Checkpoint file (!) of pre-trained weights (restore-only).")

FLAGS = tf.flags.FLAGS


class ResNet(BaseModel):
    """
    ResNet architecture for Tiny ImageNet which can be used in combination with pre-trained weights from
    https://github.com/tensorflow/models/tree/master/research/adversarial_logit_pairing.
    ResNet paper: https://arxiv.org/abs/1512.03385
    """
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

        self.pretrained_saver = None
        
        with tf.variable_scope('custom') as scope:
            self.custom_scope = scope
        
        with tf.variable_scope('resnet_v2_50', 'resnet_v2', [self.x], reuse=tf.AUTO_REUSE):
            with slim.arg_scope(self._resnet_arg_scope()):
                self.logits = self._build_model(self.x)
                self.softmax = tf.nn.softmax(self.logits, name='predictions')
        
        self._init_loss()
        self._init_accuracy()

        self.post_build_init()

    def init_saver(self) -> None:
        """
        Creates two savers: (1) for all weights (restore-and-save), (2) for pre-trained weights (restore-only).
        """
        self.saver = BaseModel._create_saver('')
        self.pretrained_saver = BaseModel._create_saver('resnet_v2_50')

    def restore(self, sess: tf.Session) -> None:
        """
        Tries to restore the weights of the model. Continues if no data is present. Tries to restore all weights first,
        then pre-trained weights only.
        :param sess: Session to restore the weights to
        """
        super().restore(sess)
        BaseModel._restore_checkpoint(self.pretrained_saver, sess, path=FLAGS.pretrained_checkpoint)

    def _build_model(self, x: tf.Tensor) -> tf.Tensor:
        """
        Builds the ResNet model graph with the TF API. This function is intentionally kept simple and sequential to
        simplify the addition of new layers.
        :param x: Input to the model, i.e. an image batch
        :return: Logits of the model
        """
        x = ResNet._first_conv(x)
        x = ResNet._v2_block(x, 'block1', base_depth=64, num_units=3, stride=2)
        x = ResNet._v2_block(x, 'block2', base_depth=128, num_units=4, stride=2)
        x = ResNet._v2_block(x, 'block3', base_depth=256, num_units=6, stride=2)
        x = ResNet._v2_block(x, 'block4', base_depth=512, num_units=3, stride=1)
        x = ResNet.batch_norm(x)
        return self.global_avg_pooling(x)

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
        self.logger_factory.add_scalar('loss', self.loss, log_frequency=25)

    def _init_accuracy(self) -> None:
        """
        Initializes the accuracy attribute. It is the percentage of correctly classified samples (value in [0,1]).
        """
        correct = tf.cast(tf.equal(tf.argmax(self.softmax, axis=1), tf.cast(self.labels, tf.int64)), dtype=tf.float32)
        self.accuracy = tf.reduce_mean(correct, name='accuracy')
        self.logger_factory.add_scalar('accuracy', self.accuracy, log_frequency=25)

    def _resnet_arg_scope(self, weight_decay: float = 0.0001) -> Dict:
        """
        :param weight_decay: Weight decay rate (0 = no decay)
        :return: Dictionary arg scope with several default arguments for conv2d, batch_norm, and max_pool2d functions
        """
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'fused': None,  # use if possible
            'is_training': self.is_training,
        }

        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=slim.variance_scaling_initializer(), activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_scope:
                    return arg_scope

    def global_avg_pooling(self, x: tf.Tensor) -> tf.Tensor:
        """
        Global average pooling along dimensions 1 and 2 followed by a conv layer.
        :param x: Input tensor
        :return: Tensor with dimensionality equal to the number of classes
        """
        x = tf.reduce_mean(x, [1, 2], name='pool5', keepdims=True)
        x = slim.conv2d(x, self.num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
        return tf.squeeze(x, [1, 2], name='spatial_squeeze')

    @staticmethod
    def batch_norm(x: tf.Tensor) -> tf.Tensor:
        """
        Wraps the slim.batch_norm method.
        :param x: Input tensor
        :return: Output tensor
        """
        return slim.batch_norm(x, activation_fn=tf.nn.relu, scope='postnorm')

    @staticmethod
    def _conv2d_same(x: tf.Tensor, num_outputs: int, kernel_size: int, stride: int, rate: int = 1, scope: str = None) \
            -> tf.Tensor:
        """
        2D convolutional layer with sizing 'SAME', i.e. input and output have the same spatial dimensionality.
        The slim.conv2d function is specified here: https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv2d
        :param x: Input tensor
        :param num_outputs: Number of output channels
        :param kernel_size: Size of the kernel
        :param stride: A sequence of N positive integers specifying the stride at which to compute output.
        :param rate: A sequence of N positive integers specifying the dilation rate to use for atrous convolution.
        :param scope: Optional scope for variable_scope
        :return: Output tensor of the convolution
        """
        if stride == 1:
            return slim.conv2d(x, num_outputs, kernel_size, stride=1, rate=rate, padding='SAME', scope=scope)

        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = tf.pad(x, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(x, num_outputs, kernel_size, stride=stride, rate=rate, padding='VALID', scope=scope)

    @staticmethod
    def _first_conv(x: tf.Tensor) -> tf.Tensor:
        """
        First conv layer of the ResNet followed by max pooling.
        :param x: Input tensor
        :return: Pooling output tensor
        """
        with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
            x = ResNet._conv2d_same(x, 64, 7, stride=2, scope='conv1')
        return slim.max_pool2d(x, [3, 3], stride=2, scope='pool1')

    @staticmethod
    def _v2_block(x: tf.Tensor, scope: str, base_depth: int, num_units: int, stride: int) -> tf.Tensor:
        """
        Adds a ResNet v2 building block to the graph. Constructs a list of parameters. Creates a 'bottleneck' element
        for each of them.
        :param x: Input tensor
        :param scope: Scope for variable_scope
        :param base_depth: Base depth
        :param num_units: Number of stacked bottleneck blocks.
        :param stride: Stride of the last bottleneck block.
        :return: Output tensor
        """
        args = [{'depth': base_depth * 4, 'depth_bottleneck': base_depth, 'stride': 1}] * (num_units - 1)
        args.append({'depth': base_depth * 4, 'depth_bottleneck': base_depth, 'stride': stride})

        with tf.variable_scope(scope, 'block', [x]):
            for i, unit in enumerate(args):
                with tf.variable_scope('unit_%d' % (i + 1), values=[x]):
                    x = ResNet._bottleneck(x, rate=1, **unit)
            x = ResNet._pooling(x)
        return x

    @staticmethod
    def _bottleneck(x: tf.Tensor, depth: int, depth_bottleneck: int, stride: int, rate: int = 1) -> tf.Tensor:
        """
        Adds a ResNet bottleneck block to the graph.
        :param x: Input tensor
        :param depth: Depth
        :param depth_bottleneck:
        :param stride: Convolution stride
        :param rate: Convolution rate
        :return: Output tensor
        """
        with tf.variable_scope(None, 'bottleneck_v2', [x]):
            depth_in = slim.utils.last_dimension(x.get_shape(), min_rank=4)
            preact = slim.batch_norm(x, activation_fn=tf.nn.relu, scope='preact')
            if depth == depth_in:
                shortcut = ResNet._pooling(x, stride, 'shortcut')
            else:
                shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None,
                                       scope='shortcut')

            res = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
            res = ResNet._conv2d_same(res, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
            res = slim.conv2d(res, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')

            return shortcut + res

    @staticmethod
    def _pooling(x: tf.Tensor, stride: int = 1, scope: Optional[str] = None) -> tf.Tensor:
        """
        Wraps slim.max_pool2d.
        :param x: Input tensor
        :param stride: Pooling stride
        :param scope: Optional variable scope name
        :return: Output tensor
        """
        if stride == 1:
            return x
        return slim.max_pool2d(x, [1, 1], stride=stride, scope=scope)
