import tensorflow as tf
import numpy as np
from resnet_base.util.lesci_experiment import LESCIExperiment
from resnet_base.util.validation import LESCIMetrics

FLAGS = tf.flags.FLAGS
BATCH_SIZE = 100

CODE_SIZES = {
    'act6_block4': [128, 256, 512, 1024],
    'act8_global_avg': [32, 128, 512],
    # 'act5_block3': [512, 1024]
}

KS = [1, 10, 20, 100]
PROJECTION_THRESHOLDS = np.linspace(0.0, 0.9, 10)
ACCURACY_CONSTRAINTS = [0.7, 0.75, 0.8]


def criterion(metrics: LESCIMetrics) -> float:
    return 1 - metrics.percentage_identity_mapped


def grid_search():
    best_experiment = None
    best_score = -1

    for lesci_pos in CODE_SIZES.keys():
        for code_size in CODE_SIZES[lesci_pos]:
            for proj_thres in PROJECTION_THRESHOLDS:
                for min_accurary in ACCURACY_CONSTRAINTS:
                    for k in KS:                                                # TODO emb_size??
                        experiment = LESCIExperiment(lesci_pos, code_size, proj_thres, k, 0, min_accurary)

                        try:
                            experiment.run()
                            experiment.print_results()

                            score = criterion(experiment.metrics)
                            if best_experiment is None or score > best_score:
                                best_experiment = experiment
                                best_score = score
                                tf.logging.info("*** This is the new best experiment! Score: {}. ***".format(score))

                        except:
                            tf.logging.error("Error occurred while executing experiment: {}"
                                             .format(experiment.experiment_description()))
