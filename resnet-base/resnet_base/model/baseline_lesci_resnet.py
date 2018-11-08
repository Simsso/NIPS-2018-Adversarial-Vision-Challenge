import tensorflow as tf
import os
from resnet_base.util.projection_metrics import projection_identity_accuracy
from resnet_base.model.baseline_resnet import BaselineResNet
from resnet_base.util.logger.factory import LoggerFactory
from resnet_base.data.tiny_imagenet_pipeline import TinyImageNetPipeline
from resnet_base.util.lesci_utils import lesci_layer

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("lesci_emb_space_file", os.path.expanduser(os.path.join('~', '.data', 'activations',
                                                                               'data_lesci_emb_space_small.mat')),
                       "Path to the file (*.mat) where embedding space values ('act_compressed') and labels ('labels') "
                       "are being stored.")
tf.flags.DEFINE_string("pca_compression_file", os.path.expanduser(os.path.join('~', '.data', 'activations',
                                                                               'pca.mat')),
                       "Path to the file (*.mat) where the PCA compression matrix ('pca_out') is stored.")

ACTIVATION_SIZES = {
    'act0_raw_input':       64*64*3,
    'act1_processed_imgs':  64*64*3,
    'act2_first_conv':      64*64*64,
    'act3_block1':          64*64*64,
    'act4_block2':          32*32*128,
    'act5_block3':          16*16*256,
    'act6_block4':          8*8*512,
    'act7_block4_postact':  8*8*512,
    'act8_global_avg':      512,
    'act9_logits':          TinyImageNetPipeline.num_classes
}


class BaselineLESCIResNet(BaselineResNet):
    """
    Represents a BaselineResNet that uses one LESCI layer.
    """

    def __init__(self, lesci_pos: str, code_size: int, proj_thres: float, k: int, emb_size: int,
                 logger_factory: LoggerFactory = None, x: tf.Tensor = None, labels: tf.Tensor = None):
        self.lesci_pos = lesci_pos
        self.code_size = code_size
        self.proj_thres = proj_thres
        self.k = k
        self.emb_size = emb_size  # n

        super().__init__(logger_factory, x, labels)

    def _build_model(self, raw_imgs: tf.Tensor) -> tf.Tensor:
        """
        Builds the ResNet model graph with the TF API. This function is intentionally kept simple and sequential to
        simplify the addition of new layers.
        :param raw_imgs: Input to the model, i.e. an image batch
        :return: Logits of the model
        """
        processed_imgs = BaselineResNet._baseline_preprocessing(raw_imgs)
        first_conv = BaselineResNet._conv2d_fixed_padding(inputs=processed_imgs, filters=64, kernel_size=3,
                                                          strides=1)

        # blocks
        block1 = BaselineResNet._block_layer(first_conv, filters=64, strides=1, is_training=self.is_training, index=1)
        block2 = BaselineResNet._block_layer(block1, filters=128, strides=2, is_training=self.is_training, index=2)
        block3 = BaselineResNet._block_layer(block2, filters=256, strides=2, is_training=self.is_training, index=3)
        block4 = BaselineResNet._block_layer(block3, filters=512, strides=2, is_training=self.is_training, index=4)

        block4_norm = BaselineResNet._batch_norm(block4, self.is_training)
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

        # add LESCI layer
        emb_shape = [self.emb_size, self.code_size]
        lesci_input = self.activations[self.lesci_pos]
        activation_size = ACTIVATION_SIZES[self.lesci_pos]
        with tf.variable_scope(self.custom_scope, auxiliary_name_scope=False):
            identity_mask, knn_label, percentage_identity_mapped = lesci_layer(lesci_input, shape=emb_shape,
                                                                               activation_size=activation_size,
                                                                               proj_thres=self.proj_thres, k=self.k)
        self.percentage_identity_mapped = percentage_identity_mapped
        self.accuracy_projection, self.accuracy_identity = projection_identity_accuracy(identity_mask, dense, knn_label,
                                                                                        labels=self.labels)
        knn_label_one_hot = tf.one_hot(knn_label, depth=TinyImageNetPipeline.num_classes)

        return tf.where(identity_mask, x=dense, y=knn_label_one_hot)


