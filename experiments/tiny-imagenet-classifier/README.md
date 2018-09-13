# Tiny ImageNet Classifier

This experiments folder contains archived experiments related to the search for a classifier for the [Tiny ImageNet data set](https://tiny-imagenet.herokuapp.com/).

The goal was to achieve classification accuracies 
* \>60% with ResNet models trained from scratch; failed, 55.81% was the best we got
* \>70% with a retrained Inception model; successful

After extensive experiments, the decision was made to drop these approaches and proceed with fine-tuning of ALP ResNet weights.
