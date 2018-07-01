# MNIST CNN Tinkering


This module package contains functionality for the following things:
 * training a CNN on MNIST and storing the weights (+ logging to TensorBoard)
 * loading the weights and analyzing the classification of linear combinations between two input samples
 * loading the weights and computing an adversarial example using a FGSM-like attack
 * generating adversarial attacks using FGSM
 * computing the layerwise perturbation cause by attacks


Start a Docker container, which mounts volumes to persist the model, outputs, and TensorBoard logging. Update the volume directory in `start.sh` (Docker requires an absolute path, here) to match this folder's location. (Previously: `/Users/timodenk/Development/nips-2018/linear-combination`)
```bash
./start.sh
```


## Model
The model is a simple CNN, taken from the [TensorFlow repository](https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/examples/tutorials/layers/cnn_mnist.py).
It is not the focus of this folder.

## Linear Combinations
The basic idea is to get two samples from the MNIST dataset (_x1_ and _x2_), where _class(x1) != class(x2)_.
We create a batch that contains _n_ images which are linear combinations of _x1_ and _x2_, given by _(1-a)*x1 + a*x2_.
For these _n_ linear combinations, we run the classification and observe how smoothly the model's predictions transition between _class(x1)_ and _class(x2)_.

We found that this method is not feasible for the detection of vulnerabilities, without significant changes.

## FGSM
To ensure that the model is vulnerable at all we have search for adversarial examples using an FGSM-like attack (and the real FGSM) and were successful.
Our first attack differs slightly from FGSM as it does not compute the gradient based on the loss but the probability output for the sample class.
