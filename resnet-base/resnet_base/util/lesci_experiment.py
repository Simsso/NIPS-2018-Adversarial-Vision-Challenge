import tensorflow as tf
import os
from resnet_base.data.tiny_imagenet_pipeline import TinyImageNetPipeline
from resnet_base.model.baseline_lesci_resnet import BaselineLESCIResNet
from resnet_base.util.validation import run_validation, LESCIMetrics

BATCH_SIZE = 100
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("mat_base_dir", os.path.expanduser('~/.data/activations/baseline/'),
                       "The directory in which all needed .mat files can be found.")


class LESCIExperiment:
    """
    Represents one hyper-parameter-tuning experiment for the BaselineLESCIResNet.
    """

    def __init__(self, lesci_pos: str, code_size: int, proj_thres: float, k: int, emb_size: int, min_accurary: float):
        """
        :param lesci_pos: the position of the LESCI-layer (e.g. 'act6-block5')
        :param code_size: the PCA-compression size for the VQ-layer
        :param proj_thres: the threshold of label-consensus needed to project a sample (in the VQ-layer)
        :param k: the number of neighbors considered by the VQ-layer
        :param emb_size: the number of embedding vectors
        :param min_accurary: the minimum mean accuracy on the validation set that this experiments needs to reach
        """
        self.lesci_pos = lesci_pos
        self.code_size = code_size
        self.proj_thres = proj_thres
        self.k = k
        self.emb_size = emb_size
        self.min_accuracy = min_accurary

        self.metrics: LESCIMetrics = None

    def run(self):
        # set up the model's flags so it uses the correct matrices corresponding to this experiment
        FLAGS.pca_compression_file = FLAGS.mat_base_dir + 'pca_{}_{}.mat'.format(self.lesci_pos, self.code_size)
        FLAGS.lesci_emb_space_file = FLAGS.mat_base_dir + 'lesci_{}_{}.mat'.format(self.lesci_pos, self.code_size)

        # set up the pipeline, create the model
        pipeline = TinyImageNetPipeline(physical_batch_size=BATCH_SIZE, shuffle=False)
        imgs, labels = pipeline.get_iterator().get_next()

        model = BaselineLESCIResNet(lesci_pos=self.lesci_pos, code_size=self.code_size, proj_thres=self.proj_thres,
                                    k=self.k, emb_size=self.emb_size, x=imgs, labels=labels)

        self.metrics = run_validation(model, pipeline, mode=tf.estimator.ModeKeys.EVAL)

    def experiment_description(self):
        return ("-- Parameters: \n" +
                "  - lesci_pos: {}\n" +
                "  - code_size: {}\n" +
                "  - proj_thres: {}\n" +
                "  - k: {}\n" +
                "  - min_accuracy: {}").format(self.lesci_pos, self.code_size, self.proj_thres, self.k,
                                               self.min_accuracy)

    def print_results(self):
        min_accuracy_reached = "YES" if self.metrics.accuracy >= self.min_accuracy else "NO"
        tf.logging.info("----- Experiment completed -----\n"
                        "{}\n"
                        "-- Results: \n"
                        "  - accuracy: {}\n"
                        "  - min_accuracy reached: {}\n"
                        "  - percentage_id_mapped: {}\n"
                        "  - accuracy_projection: {}\n".format(self.experiment_description(), self.metrics.accuracy,
                                                               min_accuracy_reached,
                                                               self.metrics.percentage_identity_mapped,
                                                               self.metrics.accuracy_projection))
