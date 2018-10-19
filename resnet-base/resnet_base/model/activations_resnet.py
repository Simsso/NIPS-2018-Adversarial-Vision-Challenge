import tensorflow as tf

from resnet_base.model.resnet import ResNet


class ActivationsResNet(ResNet):
    """
    The ActivationsResNet serves the purpose of exposing its activations in a dictionary called 'activations'. It can
    be used, e.g. to export the values of activations for given samples. Besides the dictionary attribute, this class is
    identical to the ResNet class.
    """

    def _build_model(self, img: tf.Tensor) -> tf.Tensor:
        """
        Builds the ResNet model graph with the TF API. Adds all tensors in between blocks / layers to the activations
        dictionary.
        :param img: Input to the model, i.e. an image batch
        :return: Logits of the model
        """
        first_conv = ResNet._first_conv(img)
        block1 = ResNet._v2_block(first_conv, 'block1', base_depth=64, num_units=3, stride=2)
        block2 = ResNet._v2_block(block1, 'block2', base_depth=128, num_units=4, stride=2)
        block3 = ResNet._v2_block(block2, 'block3', base_depth=256, num_units=6, stride=2)
        block4 = ResNet._v2_block(block3, 'block4', base_depth=512, num_units=3, stride=1)
        norm = ResNet.batch_norm(block4)
        pool = self.global_avg_pooling(norm)
        self.activations = {
            'act1_input': img,
            'act2_first_conv': first_conv,
            'act3_block1': block1,
            'act4_block2': block2,
            'act5_block3': block3,
            'act6_block4': block4,
            'act7_norm': norm,
            'act8_pool': pool
        }
        return pool
