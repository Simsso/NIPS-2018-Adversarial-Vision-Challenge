import os
from resnet_base.model.base_model import BaseModel
from resnet_base.data.tiny_imagenet_pipeline import TinyImageNetPipeline as Data
import tensorflow as tf
from typing import Dict, Optional

slim = tf.contrib.slim

# define flags
tf.flags.DEFINE_string("pretrained_checkpoint", os.path.expanduser('~/.models/tiny_imagenet_alp05_2018_06_26.ckpt'),
                       "Checkpoint path of pre-trained weights.")
tf.flags.DEFINE_string("custom_checkpoint", "", "Checkpoint path of custom-tuned weights.")

FLAGS = tf.flags.FLAGS


class ResNet(BaseModel):

    def __init__(self, x: tf.Tensor = None, labels: tf.Tensor = None):
        super().__init__()

        self.accuracy: tf.Tensor = None
        self.loss: tf.Tensor = None
        self.logits: tf.Tensor = None
        self.softmax: tf.Tensor = None
        self.x: tf.Tensor = tf.placeholder(tf.float32, shape=[None, Data.img_width, Data.img_height, Data.img_channels],
                                           name='x') if x is None else x
        self.is_training: tf.Tensor = tf.placeholder_with_default(False, (), 'is_training')
        self.num_classes: int = 200
        self.labels = tf.placeholder(tf.uint8, shape=[None], name='labels') if labels is None else labels

        self.pretrained_saver: tf.train.Saver = None
        self.custom_saver: tf.train.Saver = None
        with tf.variable_scope('custom') as scope:
            self.custom_scope: tf.VariableScope = scope

        with tf.variable_scope('resnet_v2_50', 'resnet_v2', [self.x], reuse=tf.AUTO_REUSE):
            with slim.arg_scope(self.resnet_arg_scope()):
                logits = self.build_model(self.x)
                self.init_outputs(logits)
        self.init_saver()
        self.init_loss()
        self.init_accuracy()

    def init_saver(self) -> None:
        # global saver (complete graph)
        self.saver = BaseModel._create_saver('')
        self.pretrained_saver = BaseModel._create_saver('resnet_v2_50')
        self.custom_saver = BaseModel._create_saver(self.custom_scope.name)

    def save(self, sess: tf.Session) -> None:
        super().load(sess)
        BaseModel._save_to_path(sess, self.pretrained_saver, self.global_step, FLAGS.pretrained_checkpoint)
        BaseModel._save_to_path(sess, self.custom_saver, self.global_step, FLAGS.custom_checkpoint)

    def load(self, sess: tf.Session) -> None:
        super().load(sess)
        BaseModel._restore_checkpoint(self.pretrained_saver, sess, FLAGS.pretrained_checkpoint)
        BaseModel._restore_checkpoint(self.custom_saver, sess, FLAGS.custom_checkpoint)

    def build_model(self, x: tf.Tensor) -> tf.Tensor:
        x = ResNet.first_conv(x)
        x = ResNet.v2_block(x, 'block1', base_depth=64, num_units=3, stride=2)
        x = ResNet.v2_block(x, 'block2', base_depth=128, num_units=4, stride=2)
        x = ResNet.v2_block(x, 'block3', base_depth=256, num_units=6, stride=2)
        x = ResNet.v2_block(x, 'block4', base_depth=512, num_units=3, stride=1)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu, scope='postnorm')
        return self.global_avg_pooling(x)  # logits

    def init_outputs(self, logits: tf.Tensor) -> None:
        self.logits = logits
        self.softmax = slim.softmax(logits, scope='predictions')

    def init_loss(self) -> None:
        labels_one_hot = tf.one_hot(self.labels, depth=Data.num_classes)
        self.loss = tf.losses.softmax_cross_entropy(labels_one_hot, self.logits)

    def init_accuracy(self) -> None:
        correct = tf.cast(tf.equal(tf.argmax(self.softmax, axis=1), tf.cast(self.labels, tf.int64)), dtype=tf.float32)
        self.accuracy = tf.reduce_mean(correct, name='accuracy')

    def resnet_arg_scope(self, weight_decay: float = 0.0001) -> Dict:
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
        x = tf.reduce_mean(x, [1, 2], name='pool5', keepdims=True)
        x = slim.conv2d(x, self.num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
        return tf.squeeze(x, [1, 2], name='SpatialSqueeze')

    @staticmethod
    def conv2d_same(x: tf.Tensor, num_outputs: int, kernel_size: int, stride: int, rate: int = 1, scope: str = None)\
            -> tf.Tensor:
        if stride == 1:
            return slim.conv2d(x, num_outputs, kernel_size, stride=1, rate=rate, padding='SAME', scope=scope)

        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = tf.pad(x, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(x, num_outputs, kernel_size, stride=stride, rate=rate, padding='VALID', scope=scope)

    @staticmethod
    def first_conv(x: tf.Tensor) -> tf.Tensor:
        with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
            x = ResNet.conv2d_same(x, 64, 7, stride=2, scope='conv1')
        return slim.max_pool2d(x, [3, 3], stride=2, scope='pool1')

    @staticmethod
    def v2_block(x: tf.Tensor, scope: str, base_depth: int, num_units: int, stride: int) -> tf.Tensor:
        args = [{'depth': base_depth * 4, 'depth_bottleneck': base_depth, 'stride': 1}] * (num_units - 1)
        args.append({'depth': base_depth * 4, 'depth_bottleneck': base_depth, 'stride': stride})

        with tf.variable_scope(scope, 'block', [x]):
            block_stride = 1
            for i, unit in enumerate(args):
                with tf.variable_scope('unit_%d' % (i + 1), values=[x]):
                    x = ResNet.bottleneck(x, rate=1, **unit)
            x = ResNet.pooling(x, block_stride)
        return x

    @staticmethod
    def bottleneck(inputs: tf.Tensor, depth: int, depth_bottleneck: int, stride: int, rate: int = 1) -> tf.Tensor:
        with tf.variable_scope(None, 'bottleneck_v2', [inputs]):
            depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
            preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
            if depth == depth_in:
                shortcut = ResNet.pooling(inputs, stride, 'shortcut')
            else:
                shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None,
                                       scope='shortcut')

            res = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
            res = ResNet.conv2d_same(res, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
            res = slim.conv2d(res, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')

            return shortcut + res

    @staticmethod
    def pooling(inputs: tf.Tensor, factor: int, scope: Optional[str] = None) -> tf.Tensor:
        if factor == 1:
            return inputs
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)
