from resnet_base.model.lesci_resnet import LESCIResNet
import tensorflow as tf

from foolbox.models import TensorFlowModel
from resnet_base.util.logger.factory import LoggerFactory


class SubmittableResNet(LESCIResNet):
    """
    This is a simple wrapper around the ResNet model which offers a function that converts the model to a
    foolbox model which can easily be submitted to the challenge website.
    """

    def __init__(self, logger_factory: LoggerFactory = None, x: tf.Tensor = None, labels: tf.Tensor = None):
        super().__init__(logger_factory=logger_factory, x=x, labels=labels)

    def get_foolbox_model(self) -> TensorFlowModel:
        """
        Returns an instance of a foolbox Model (more specifically, a TensorFlowModel) which is configured to use
        the x-placeholder and the logits of this model. It specifies pre-processing settings as expected by the model.
        """
        images = self.x
        logits = self.logits

        sess = tf.get_default_session()
        self.restore(sess)

        # this pre-processing setup normalizes the image's values to be in [-1, 1], as expected by the model
        fmodel = TensorFlowModel(images, logits, bounds=(0, 255), preprocessing=(127.5, 127.5))
        return fmodel
