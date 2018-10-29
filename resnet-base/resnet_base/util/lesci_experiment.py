import tensorflow as tf

FLAGS = tf.flags.FLAGS


class LESCIExperiment:
    """
    Represents one hyper-parameter-tuning experiment for the BaselineLESCIResNet.
    """

    def __init__(self, lesci_pos: str, compression: int, proj_thres: float, k: int, min_accurary: float):
        """
        :param lesci_pos: the position of the LESCI-layer (e.g. 'act6-block5')
        :param compression: the PCA-compression rate of the VQ-layer
        :param proj_thres: the threshold of label-consensus needed to project a sample (in the VQ-layer)
        :param k: the number of neighbors considered by the VQ-layer
        :param min_accurary: the minimum mean accuracy on the validation set that this experiments needs to reach
        """
        self.lesci_pos = lesci_pos
        self.compression = compression
        self.proj_thres = proj_thres
        self.k = k
        self.min_accuracy = min_accurary

    def run(self):
        pass
