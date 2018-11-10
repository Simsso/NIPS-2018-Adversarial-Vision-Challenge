# NIPS 2018 Adversarial Vision Challenge "Robust Model Track"
_[Timo I. Denk](https://timodenk.com/), [Florian Pfisterer](https://twitter.com/FlorianPfi), [Samed Guener](https://twitter.com/samedguener)  
(published in November 2018)_

## Abstract
This repository contains code, documents, and deployment configuration files, related to our participation in the 2018 NIPS Adversarial Vision Challenge "Robust Model Track".  
We implemented a technique called a _LESCI-layer_ which is based on vector quantization (VQ) and supposed to increase the robustness of a neural network classifier.
It compresses the representation at a certain layer with a matrix computed using PCA on representations induced by correctly classified training samples at this layer.
The compressed vector is being compared to an embedding space; and replaced with an embedding vector if a certain percentage of the _k_ most similar vectors belong to the same output label.  
In the current configuration, our method did not increase the robustness of the ResNet-based classifier for Tiny ImageNet, as measured by the challenge, presumably because it comes with a decrease in classification accuracy.
We have documented our approach formally in [this PDF](./docs/article/article.pdf).


## Background

The annual NIPS conference has a [competition track](https://nips.cc/Conferences/2018/CompetitionTrack).
We have participated in the [Adversarial Vision Challenge "Robust Model Track"](https://www.crowdai.org/challenges/adversarial-vision-challenge):
> The overall goal of this challenge is to facilitate measurable progress towards robust machine vision models and more generally applicable adversarial attacks. As of right now, modern machine vision algorithms are extremely susceptible to small and almost imperceptible perturbations of their inputs (so-called adversarial examples). This property reveals an astonishing difference in the information processing of humans and machines and raises security concerns for many deployed machine vision systems like autonomous cars. Improving the robustness of vision algorithms is thus important to close the gap between human and machine perception and to enable safety-critical applications.

## Team

We are three CS students from Germany who worked on the NIPS project in their leisure time.

![team picture](https://user-images.githubusercontent.com/6556307/48194510-88171080-e34d-11e8-8d9a-705f82b6a50d.png)

These were our responsibilities:

* **Timo Denk** _(left; [@simsso](https://github.com/simsso))_: Team lead, architectural decisions, Python development, ML research and ideas.
* **Florian Pfisterer** _(middle; [@florianpfisterer](https://github.com/florianpfisterer))_: Architectural decisions, Python development, ML research and ideas.
* **Samed Güner** _(right; [@doktorgibson](https://github.com/doktorgibson))_: Training pipeline design, cloud administration, pipeline implementation.

## Repository

This repository is an integral component of our work and served the following purposes:
* **Code**. The repository contains the entire commit history of our code. During development it has proven to be an effective way of catching up with the commits of other team members.
* **Project management tool**. We used issues quite extensively to keep track of [work items](https://github.com/Simsso/NIPS-2018-Adversarial-Vision-Challenge/issues?q=label%3Awork-item+) and [meetings](https://github.com/Simsso/NIPS-2018-Adversarial-Vision-Challenge/issues?q=label%3Ameeting). For each meeting we took notes of assignments and documented the progress we had made.
* **Knowledge base**. The repository's [wiki](https://github.com/Simsso/NIPS-2018-Adversarial-Vision-Challenge/wiki) contains enduring ideas and documentation, such as [how our pipeline is set up](https://github.com/Simsso/NIPS-2018-Adversarial-Vision-Challenge/wiki/Training-Pipeline), which [papers we consider relevant](https://github.com/Simsso/NIPS-2018-Adversarial-Vision-Challenge/wiki/Reading-List), or [how we name our commits](https://github.com/Simsso/NIPS-2018-Adversarial-Vision-Challenge/wiki/Repository-Conventions), just to name a few.
* **Review**. Every contribution to the `master` branch had to be reviewed. In total we opened [more than 25 pull requests](https://github.com/Simsso/NIPS-2018-Adversarial-Vision-Challenge/pulls?q=is%3Apr); some of which received more than 30 comments.
* **DevOps**. We set up webhooks to the Google Cloud Platform to be able to automatically spin up new instances for training, once a commit was flagged with a certain tag.

## Codebase
Our codebase consists of two Python modules, namely `resnet_base` and `vq_layer`. In addition to that we publish an `experiments` folder which contains _dirty_ code that was written for the sake of testing ideas. This section mentions some specifics and references the actual documentation. The class diagrams were generated with `pyreverse`. TODO(florianpfisterer): update resnet_base to match the new name and update links accordingly.

### VQ-Layer
The `vq_layer` module contains TensorFlow (TF) implementations of our vector quantization ideas. Following the TF API, that is a [number of functions](https://github.com/Simsso/NIPS-2018-Adversarial-Vision-Challenge/blob/master/vq-layer/vq_layer/vq_layer.py) which work with `tf.Tensor` objects. The features as well as install instructions can be found in the [README file of the module](https://github.com/Simsso/NIPS-2018-Adversarial-Vision-Challenge/blob/master/vq-layer/README.md).

We prioritized a good test coverage to ensure the proper functioning of the module. Each of the test classes covers one specific aspect (described in a comment) of the module. The test classes share some functionality, e.g. graph reset, session creation, and random seed, which we have placed in the `TFTestCase` class.

![vq_layer class diagram](https://user-images.githubusercontent.com/6556307/48197469-1c857100-e356-11e8-9469-2451c8e38654.png)  
_Fig.: Class diagram of the module `vq_layer`. It shows the test classes which inherit from `TFTestCase`. Each class is responsible for testing a specific aspect for which it implements a plurality of test cases (methods)._

### ResNet Base

The `resnet_base` module contains our approaches to developing a more robust classifier for the Tiny ImageNet dataset. The documentation can be found [in the README file](https://github.com/Simsso/NIPS-2018-Adversarial-Vision-Challenge/blob/master/resnet-base/README.md).

Our basic idea was to be able to try out new things by inheriting from some `Model` class and overriding its graph construction method. The new method would then contain some special features that we want to test. This idea is reflected in the class diagram below. 

The **`BaseModel`** contains fundamental methods and attributes that all our ML models need. For instance an epoch counter or functionality for saving and restoring weights. The two inheriting classes are two ResNet implementations that can restore pre-trained weights. 

**`BaselineResNet`** is designed to work with baseline weights [provided by the challenge organizers](https://gitlab.crowdai.org/adversarial-vision-challenge/resnet18_model_baseline/tree/master/resnet18/checkpoints/model), while **`ResNet`** works with ["ALP-trained ResNet-v2-50" weights](https://github.com/tensorflow/models/tree/master/research/adversarial_logit_pairing#pre-trained-models).

The classes inheriting from `BaselineResNet` and `ResNet` are our experiments: **`BaselineLESCIResNet`**, **`LESCIResNet`**, **`PCAResNet`**, **`ParallelVQResNet`**, **`VQResNet`**, and **`ActivationsResNet`**. They are typically using the functions provided by the `vq_layer` module.

Our **input pipeline** provides the models with images from the Tiny ImageNet dataset. It follows [the official recommendation](https://www.tensorflow.org/guide/datasets) by using TF's `tf.data` API. The code is split into more generic functions, which might be reused in pipelines for other datasets (`BasePipeline` class), and the code specific to the Tiny ImageNet dataset (`TinyImageNetPipeline` class), for instance reading label text files or image augmentation.

Our **logging** is quite comprehensive. Because we accumulate gradients over several _physical batches_, we cannot use the plain `tf.summary` API and have to accumulate scalars and histograms in order to create a `tf.Summary` object manually. This functionality is placed in `Logger`, `Accumulator`, and inheriting classes.

![resnet_base class diagram](https://user-images.githubusercontent.com/6556307/48197117-09be6c80-e355-11e8-8a97-7e2b43edc8e6.png)  
_Fig.: Class diagram of the module `resnet_base`. Accumulators are on the left, the different models are in the middle, the pipeline and misc. is on the right._


### Experiments

Our [experiments](https://github.com/Simsso/NIPS-2018-Adversarial-Vision-Challenge/tree/master/experiments) are a collection of Python scripts, MATLAB files, and Jupyter notebooks. Some highlights are:
* A visualization of the embedding space training: [experiments/vq-layer/003-embedding-space-training.ipynb](https://github.com/Simsso/NIPS-2018-Adversarial-Vision-Challenge/blob/master/experiments/vq-layer/003-embedding-space-training.ipynb)
* Our implementation of the fast gradient sign method (FGSM): [experiments/mnist/src/attack.py](https://github.com/Simsso/NIPS-2018-Adversarial-Vision-Challenge/blob/master/experiments/mnist/src/attack.py).  
![image](https://user-images.githubusercontent.com/6556307/41718032-fbbc0c76-755b-11e8-902e-f752438935d6.png) ![image](https://user-images.githubusercontent.com/6556307/41718050-0fe0cd68-755c-11e8-9e57-864f75e0daef.png)  
_Fig.: Gradient (right) of loss wrt. input for a sample (left) from the MNIST dataset._

## Training Pipeline
Our DevOps unit (Samed) has set up a training pipeline that simplifies the empirical evaluation of ML ideas. For the researcher, triggering a training run is a simple as tagging a commit and pushing it. The tag triggers a pipeline which creates a new virtual machine (VM) on the Google Cloud Platform (GCP). The VM is configured to have a GPU and to run the training job (Python files). The results (e.g. model weights and logged metrics) were streamed to a persistent storage which the ML researcher could access through the GCP user interface and a TensorBoard instance which we kept running.

More details about the pipeline can be found [here](https://github.com/Simsso/NIPS-2018-Adversarial-Vision-Challenge/wiki/Training-Pipeline) and an analysis of GCP's capabilities (from our perspective) is written down [here](https://github.com/Simsso/NIPS-2018-Adversarial-Vision-Challenge/wiki/Google-Cloud-Platform).

## Results
Our final submission was intended to be a pre-trained ResNet (baseline supplied by the challenge) which uses a LESCI-layer at a level in the network that gives a good balance between robustness and accuracy 
(more about our reasoning behind this in our [PDF article](./docs/article/article.pdf)).

Computing the PCA on the activations from early layers turned out to be computationally infeasible in terms of memory requirement, which is why we had to constrain our hyperparameter search to only one position late in the network where the dimension of an
activation was only 512 (instead of 131,072 up to 262,144 in higher layers). 

Unfortunately, this hyperparameter grid search gave us no combination of parameters that resulted in an accuracy of more than 50.0% and a good percentage of inputs that were projected at the same time (calculated based on the Tiny ImageNet validation set).

Future work should focus on inserting a LESCI-layer at an earlier layer in the network, which we were not able to do for a lack of computational resources.
