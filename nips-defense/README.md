# NIPS Defense
_This folder contains a self-contained ResNet model which provides the basis for our experiments._

At its core, it consists of two ResNet models, namely 
[ALP-ResNet model](https://github.com/tensorflow/models/tree/master/research/adversarial_logit_pairing) and [NIPS challenge baseline ResNet](https://gitlab.crowdai.org/florianpfisterer/resnet18_model_baseline),
incorporated into a popular [TensorFlow project template](https://github.com/MrGemy95/Tensorflow-Project-Template)
(which we have adapted to our needs). The two ResNet models are classes which we extend in sub-classes that add new functionality to the models, trying to increase robustness.

### Highlights
The following features are implemented:
* restore the pre-trained [ALP weights](https://github.com/tensorflow/models/tree/master/research/adversarial_logit_pairing); class `ResNet`.
* restore the pre-trained [challenge baseline weights](https://gitlab.crowdai.org/florianpfisterer/resnet18_model_baseline); class `BaselineResNet`.
* load the [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) training and validation set
* channel training and validation data via a `tf.data` input pipeline, apply pre-processing
* evaluate the model on the validation set (using [`eval.py`](nips_defense/mains/eval.py))
* support multiple savers to restore and save different parts of the graph, namely pre-trained, custom and global weights
* perform training using [`resnet_trainer.py`](nips_defense/trainer/resnet_trainer.py)
(can be tested with [`test_train.py`](nips_defense/mains/test_train.py))
* validate new techniques against foolbox attacks [`score.py`](nips_defense/mains/score.py)


### Flags
| Flag | Sample | Description |
| --- | --- | --- |
| `data_dir` | ~/.data/tiny-imagenet-200 | Path of the Tiny ImageNet dataset folder |
| `enable_train_augmentation` | False | Whether to enable image augmentation for training samples |
| `physical_batch_size` | 32 | Number of samples per batch that is fed through the GPU at once. |
| `virtual_batch_size_factor` | 8 | "Number of batches per weight update." |
| `save_dir` | ~/.models/output | Checkpoint directory of the complete graph's variables. Used both to restore (if available) and to save the model. |
| `name` | testmodel1-dropout0.5 | The name of the model (may contain hyperparameter information), used when saving the model. |
| `learning_rate` | 1 | The learning rate used for training. |
| `num_epochs` | 100 | The number of epochs for which training is performed. |
| `train_log_dir` | ../tf_logs/train | The directory used to save the training summaries. |
| `val_log_dir` | ../tf_logs/val | The directory used to save the validation summaries. |
| `pca_mat_file` | ~/.data/activations/pca.mat | Path to the file (*.mat) where the PCA matrices are being stored. |
| `activations_export_file` | ~/.data/activations/data_100k_act5_block3 | File to export the activations to, without file extension. |
| `baseline_checkpoint` | ~/.models/resnet-baseline/model.ckpt | Checkpoint file for the baseline ResNet model. |
| `lesci_emb_space_file` | ~/.data/activations/data_lesci_emb_space_small.mat | Path to the file (*.mat) where embedding space values ('act_compressed') and labels ('labels') are being stored. |
| `pca_compression_file` | ~/.data/activations/pca.mat | Path to the file (*.mat) where the PCA compression matrix ('pca_out') is stored. |

### Local Execution
Install with 
```bash
python setup.py install
```
Point the `__main__.py` file to the desired Python script in the `main` folder and run using
```bash
python -m nips_defense
```
