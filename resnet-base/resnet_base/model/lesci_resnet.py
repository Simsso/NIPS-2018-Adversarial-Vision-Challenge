import os
from resnet_base.model.resnet import ResNet
import tensorflow as tf
from resnet_base.data.tiny_imagenet_pipeline import TinyImageNetPipeline
from resnet_base.util.projection_metrics import projection_identity_accuracy
from resnet_base.util.lesci_utils import lesci_layer

tf.flags.DEFINE_string("lesci_emb_space_file", os.path.expanduser(os.path.join('~', '.data', 'activations',
                                                                               'data_lesci_emb_space_small.mat')),
                       "Path to the file (*.mat) where embedding space values ('act_compressed') and labels ('labels') "
                       "are being stored.")
tf.flags.DEFINE_string("pca_compression_file", os.path.expanduser(os.path.join('~', '.data', 'activations',
                                                                               'pca.mat')),
                       "Path to the file (*.mat) where the PCA compression matrix ('pca_out') is stored.")
FLAGS = tf.flags.FLAGS


class LESCIResNet(ResNet):
    """
    ResNet version that can restore ALP weights and contains a LESCI layer.
    """
    def _build_model(self, x: tf.Tensor) -> tf.Tensor:
        x = ResNet._first_conv(x)  # 16x16x64
        x = ResNet._v2_block(x, 'block1', base_depth=64, num_units=3, stride=2)
        x = ResNet._v2_block(x, 'block2', base_depth=128, num_units=4, stride=2)
        x = ResNet._v2_block(x, 'block3', base_depth=256, num_units=6, stride=2)  # 2x2x1024

        with tf.variable_scope(self.custom_scope, auxiliary_name_scope=False):
            identity_mask, knn_label, percentage_identity_mapped = lesci_layer(x, shape=[74246, 64],
                                                                               activation_size=2*2*1024, proj_thres=0.5,
                                                                               k=10)
        self.percentage_identity_mapped = percentage_identity_mapped

        x = ResNet._v2_block(x, 'block4', base_depth=512, num_units=3, stride=1)
        x = ResNet.batch_norm(x)
        resnet_out = self.global_avg_pooling(x)
        knn_label_one_hot = tf.one_hot(knn_label, depth=TinyImageNetPipeline.num_classes)

        self.accuracy_projection, self.accuracy_identity = projection_identity_accuracy(identity_mask, resnet_out,
                                                                                        knn_label, labels=self.labels)
        return tf.where(identity_mask, x=resnet_out, y=knn_label_one_hot)
