import tensorflow as tf

from data.tiny_imagenet_pipeline import TinyImageNetPipeline
from model.baseline_lesci_resnet import BaselineLESCIResNet
from util.validation import run_validation

BATCH_SIZE = 100
FLAGS = tf.flags.FLAGS


class LESCIExperiment:
    """
    Represents one hyper-parameter-tuning experiment for the BaselineLESCIResNet.
    """

    def __init__(self, lesci_pos: str, code_size: int, proj_thres: float, k: int, min_accurary: float):
        """
        :param lesci_pos: the position of the LESCI-layer (e.g. 'act6-block5')
        :param code_size: the PCA-compression size for the VQ-layer
        :param proj_thres: the threshold of label-consensus needed to project a sample (in the VQ-layer)
        :param k: the number of neighbors considered by the VQ-layer
        :param min_accurary: the minimum mean accuracy on the validation set that this experiments needs to reach
        """
        self.lesci_pos = lesci_pos
        self.code_size = code_size
        self.proj_thres = proj_thres
        self.k = k
        self.min_accuracy = min_accurary

        self.metrics = None

    def run(self):
        pipeline = TinyImageNetPipeline(physical_batch_size=BATCH_SIZE, shuffle=False)
        imgs, labels = pipeline.get_iterator().get_next()

        model = BaselineLESCIResNet(lesci_pos=self.lesci_pos, code_size=self.code_size, proj_thres=self.proj_thres,
                                    k=self.k, x=imgs, labels=labels)
        self.metrics = run_validation(model, pipeline, mode=tf.estimator.ModeKeys.EVAL)
