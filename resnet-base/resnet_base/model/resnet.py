import collections

from resnet_base.model.base_model import BaseModel
import tensorflow as tf

slim = tf.contrib.slim


class ResNet(BaseModel):

    def __init__(self, checkpoint_dir: str):
        self.x: tf.Tensor = None
        self.is_training: tf.Tensor = tf.placeholder_with_default(False, (), 'is_training')
        self.num_classes: int = 200
        super().__init__(checkpoint_dir)

    def init_saver(self) -> None:
        var_list = tf.contrib.framework.get_variables_to_restore()
        self.saver = tf.train.Saver(var_list=var_list)

    def build_model(self) -> None:
        with tf.variable_scope('resnet_v2_50', 'resnet_v2', [self.x], reuse=tf.AUTO_REUSE) as sc:
            blocks = [
                ResNet.v2_block('block1', base_depth=64, num_units=3, stride=2),
                ResNet.v2_block('block2', base_depth=128, num_units=4, stride=2),
                ResNet.v2_block('block3', base_depth=256, num_units=6, stride=2),
                ResNet.v2_block('block4', base_depth=512, num_units=3, stride=1),
            ]
            end_points_collection = sc.original_name_scope + '_end_points'
            # with slim.arg_scope([slim.conv2d, bottleneck,
            #                     resnet_utils.stack_blocks_dense],
            #                    outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm], is_training=self.is_training):
                net = self.x
                # We do not include batch normalization or activation functions in
                # conv1 because the first ResNet unit will perform these. Cf.
                # Appendix of [2].
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=None, normalizer_fn=None):
                    net = ResNet.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                net = ResNet.stack_blocks_dense(net, blocks)
                # This is needed because the pre-activation variant does not have batch
                # normalization or activation functions in the residual unit output. See
                # Appendix of [2].
                net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                # Global average pooling.
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                end_points['global_pool'] = net
                net = slim.conv2d(net, self.num_classes, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope='logits')
                end_points[sc.name + '/logits'] = net
                net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                end_points[sc.name + '/spatial_squeeze'] = net
                end_points['predictions'] = slim.softmax(net, scope='predictions')

    def init_loss(labels: tf.Tensor, logits: tf.Tensor) -> None:
        labels_one_hot = tf.one_hot(labels, depth=data.NUM_CLASSES)
        cross_entropy_loss = tf.losses.softmax_cross_entropy(labels_one_hot, logits)

    def get_accuracy(labels: tf.Tensor, softmax: tf.Tensor) -> tf.Tensor:
        correct = tf.cast(tf.equal(tf.argmax(softmax, axis=1), tf.cast(labels, tf.int64)), dtype=tf.float32)
        return tf.reduce_mean(correct, name='accuracy')

    @staticmethod
    def conv2d_same(inputs: tf.Tensor, num_outputs: int, kernel_size: int, stride: int, rate: int = 1,
                    scope: str = None) -> tf.Tensor:
        if stride == 1:
            return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
                               padding='SAME', scope=scope)
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            inputs = tf.pad(inputs,
                            [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
            return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                               rate=rate, padding='VALID', scope=scope)

    @staticmethod
    def v2_block(scope, base_depth, num_units, stride):
        return Block(scope, ResNet.bottleneck, [{
            'depth': base_depth * 4,
            'depth_bottleneck': base_depth,
            'stride': 1
        }] * (num_units - 1) + [{
            'depth': base_depth * 4,
            'depth_bottleneck': base_depth,
            'stride': stride
        }])

    @staticmethod
    @slim.add_arg_scope
    def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
                   outputs_collections=None, scope=None):
        """Bottleneck residual unit variant with BN before convolutions.
        This is the full preactivation residual unit variant proposed in [2]. See
        Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
        variant which has an extra bottleneck layer.
        When putting together two consecutive ResNet blocks that use this unit, one
        should use stride = 2 in the last unit of the first block.
        Args:
          inputs: A tensor of size [batch, height, width, channels].
          depth: The depth of the ResNet unit output.
          depth_bottleneck: The depth of the bottleneck layers.
          stride: The ResNet unit's stride. Determines the amount of downsampling of
            the units output compared to its input.
          rate: An integer, rate for atrous convolution.
          outputs_collections: Collection to add the ResNet unit output.
          scope: Optional variable_scope.
        Returns:
          The ResNet unit's output.
        """
        with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
            depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
            preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
            if depth == depth_in:
                shortcut = ResNet.subsample(inputs, stride, 'shortcut')
            else:
                shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                       normalizer_fn=None, activation_fn=None,
                                       scope='shortcut')

            residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                                   scope='conv1')
            residual = ResNet.conv2d_same(residual, depth_bottleneck, 3, stride,
                                          rate=rate, scope='conv2')
            residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='conv3')

            output = shortcut + residual

            return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)

    @staticmethod
    @slim.add_arg_scope
    def stack_blocks_dense(net, blocks, output_stride=None,
                           store_non_strided_activations=False,
                           outputs_collections=None):
        """Stacks ResNet `Blocks` and controls output feature density.
        First, this function creates scopes for the ResNet in the form of
        'block_name/unit_1', 'block_name/unit_2', etc.
        Second, this function allows the user to explicitly control the ResNet
        output_stride, which is the ratio of the input to output spatial resolution.
        This is useful for dense prediction tasks such as semantic segmentation or
        object detection.
        Most ResNets consist of 4 ResNet blocks and subsample the activations by a
        factor of 2 when transitioning between consecutive ResNet blocks. This results
        to a nominal ResNet output_stride equal to 8. If we set the output_stride to
        half the nominal network stride (e.g., output_stride=4), then we compute
        responses twice.
        Control of the output feature density is implemented by atrous convolution.
        Args:
          net: A `Tensor` of size [batch, height, width, channels].
          blocks: A list of length equal to the number of ResNet `Blocks`. Each
            element is a ResNet `Block` object describing the units in the `Block`.
          output_stride: If `None`, then the output will be computed at the nominal
            network stride. If output_stride is not `None`, it specifies the requested
            ratio of input to output spatial resolution, which needs to be equal to
            the product of unit strides from the start up to some level of the ResNet.
            For example, if the ResNet employs units with strides 1, 2, 1, 3, 4, 1,
            then valid values for the output_stride are 1, 2, 6, 24 or None (which
            is equivalent to output_stride=24).
          store_non_strided_activations: If True, we compute non-strided (undecimated)
            activations at the last unit of each block and store them in the
            `outputs_collections` before subsampling them. This gives us access to
            higher resolution intermediate activations which are useful in some
            dense prediction problems but increases 4x the computation and memory cost
            at the last unit of each block.
          outputs_collections: Collection to add the ResNet block outputs.
        Returns:
          net: Output tensor with stride equal to the specified output_stride.
        Raises:
          ValueError: If the target output_stride is not valid.
        """
        # The current_stride variable keeps track of the effective stride of the
        # activations. This allows us to invoke atrous convolution whenever applying
        # the next residual unit would result in the activations having stride larger
        # than the target output_stride.
        current_stride = 1

        # The atrous convolution rate parameter.
        rate = 1

        for block in blocks:
            with tf.variable_scope(block.scope, 'block', [net]) as sc:
                block_stride = 1
                for i, unit in enumerate(block.args):
                    if store_non_strided_activations and i == len(block.args) - 1:
                        # Move stride from the block's last unit to the end of the block.
                        block_stride = unit.get('stride', 1)
                        unit = dict(unit, stride=1)

                    with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                        # If we have reached the target output_stride, then we need to employ
                        # atrous convolution with stride=1 and multiply the atrous rate by the
                        # current unit's stride for use in subsequent layers.
                        if output_stride is not None and current_stride == output_stride:
                            net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
                            rate *= unit.get('stride', 1)

                        else:
                            net = block.unit_fn(net, rate=1, **unit)
                            current_stride *= unit.get('stride', 1)
                            if output_stride is not None and current_stride > output_stride:
                                raise ValueError('The target output_stride cannot be reached.')

                # Collect activations at the block's end before performing subsampling.
                net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

                # Subsampling of the block's output activations.
                if output_stride is not None and current_stride == output_stride:
                    rate *= block_stride
                else:
                    net = ResNet.subsample(net, block_stride)
                    current_stride *= block_stride
                    if output_stride is not None and current_stride > output_stride:
                        raise ValueError('The target output_stride cannot be reached.')

        if output_stride is not None and current_stride != output_stride:
            raise ValueError('The target output_stride cannot be reached.')

        return net

    @staticmethod
    def subsample(inputs, factor, scope=None):
        """Subsamples the input along the spatial dimensions.
        Args:
          inputs: A `Tensor` of size [batch, height_in, width_in, channels].
          factor: The subsampling factor.
          scope: Optional variable_scope.
        Returns:
          output: A `Tensor` of size [batch, height_out, width_out, channels] with the
            input, either intact (if factor == 1) or subsampled (if factor > 1).
        """
        if factor == 1:
            return inputs
        else:
            return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.
    Its parts are:
        scope: The scope of the `Block`.
        unit_fn: The ResNet unit function which takes as input a `Tensor` and
            returns another `Tensor` with the output of the ResNet unit.
        args: A list of length equal to the number of units in the `Block`. The list
            contains one (depth, depth_bottleneck, stride) tuple for each unit in the
            block to serve as argument to unit_fn.
    """
