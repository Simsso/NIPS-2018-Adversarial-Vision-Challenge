import tensorflow as tf
import numpy as np
from resnet_base.util.lesci_experiment import LESCIExperiment
from resnet_base.util.validation import LESCIMetrics

FLAGS = tf.flags.FLAGS
BATCH_SIZE = 100

CODE_SIZES = {
    'act6_block4': [128, 256, 512, 1024],
    'act8_global_avg': [32, 128, 512],
    'act5_block3': [512, 1024]
}

KS = [1, 10, 20, 100]
PROJECTION_THRESHOLDS = np.linspace(0.0, 0.9, 10)
ACCURACY_CONSTRAINTS = [0.7, 0.75, 0.8]


def criterion(metrics: LESCIMetrics) -> float:
    return 1 - metrics.percentage_identity_mappe


def grid_search():
    for lesci_pos in CODE_SIZES.keys():
        for code_size in CODE_SIZES[lesci_pos]:
            for proj_thres in PROJECTION_THRESHOLDS:
                for min_accurary in ACCURACY_CONSTRAINTS:
                    for k in KS:
                        experiment = LESCIExperiment(lesci_pos, code_size, proj_thres, k, min_accurary)
                        experiment.run()

                        # TODO manage results
