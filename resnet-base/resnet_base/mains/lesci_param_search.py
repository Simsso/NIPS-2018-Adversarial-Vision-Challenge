import tensorflow as tf
import numpy as np
import time
from resnet_base.util.lesci_experiment import LESCIExperiment
from resnet_base.util.validation import LESCIMetrics

tf.logging.set_verbosity(tf.logging.DEBUG)
FLAGS = tf.flags.FLAGS
BATCH_SIZE = 100

CODE_SIZES = {
    # 'act6_block4': [128, 256, 512, 1024],
    'act8_global_avg': [32, 128, 512],
    # 'act5_block3': [512, 1024]
}

KS = [1, 10, 20, 100]
PROJECTION_THRESHOLDS = np.linspace(0.0, 0.9, 10)
ACCURACY_CONSTRAINTS = [0.7]
EMB_SIZES = {
    'act6_block4': 82480,
    'act8_global_avg': 86706
}


def criterion(metrics: LESCIMetrics) -> float:
    return 1 - metrics.percentage_identity_mapped


def grid_search():
    best_experiment = None
    best_score = -1

    for lesci_pos in CODE_SIZES.keys():
        for code_size in CODE_SIZES[lesci_pos]:
            for proj_thres in PROJECTION_THRESHOLDS:
                for min_accurary in ACCURACY_CONSTRAINTS:
                    for k in KS:
                        emb_size = EMB_SIZES[lesci_pos]
                        experiment = LESCIExperiment(lesci_pos, code_size, proj_thres, k, emb_size, min_accurary)

                        try:
                            experiment.run()
                            experiment.print_results()

                            score = criterion(experiment.metrics)
                            accuracy = experiment.metrics.accuracy
                            if best_experiment is None or (score > best_score and accuracy > min_accurary):
                                best_experiment = experiment
                                best_score = score
                                tf.logging.info("*** This is the new best experiment! Score: {}. ***".format(score))
                        except Exception as error:
                            tf.logging.error("Error occurred while executing experiment: {}, \n{}"
                                             .format(error, experiment.experiment_description()))


def main(args):
    tf.logging.info("Starting grid search...")
    start = time.time()
    grid_search()
    end = time.time()
    tf.logging.info("Completed grid search. Took {:.1f} seconds".format((end - start)))


if __name__ == '__main__':
    tf.app.run()
