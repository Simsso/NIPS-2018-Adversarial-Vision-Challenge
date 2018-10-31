# NIPS 2018 Adversarial Vision Challenge "Robust Model Track"
_[Timo I. Denk](https://timodenk.com/), [Florian Pfisterer](https://twitter.com/florianpfi), [Samed Guener](https://twitter.com/samedguener)  
(published in November 2018)_

**Abstract**. This repository contains code, documents, and deployment configuration files, related to our participation in the 2018 NIPS Adversarial Vision Challenge "Robust Model Track".  
We have implemented a technique called LESCI which is supposed to increase the robustness of a neural network classifier. It compresses the output of a layer with a matrix computed using PCA on vectors induced by correctly classified training samples. The compressed vector is being compared to an embedding space; and replaced with contained vectors if a certain percentage of the _k_ most similar vectors belongs to the same output label.  
Our method did not increase the robustness of the model as measured by the challenge, presumably because it comes with a decrease in classification accuracy.


## Background

The annual NIPS conference has a [competition track](https://nips.cc/Conferences/2018/CompetitionTrack). We have participated in the [Adversarial Vision Challenge "Robust Model Track"](https://www.crowdai.org/challenges/adversarial-vision-challenge):
> The overall goal of this challenge is to facilitate measurable progress towards robust machine vision models and more generally applicable adversarial attacks. As of right now, modern machine vision algorithms are extremely susceptible to small and almost imperceptible perturbations of their inputs (so-called adversarial examples). This property reveals an astonishing difference in the information processing of humans and machines and raises security concerns for many deployed machine vision systems like autonomous cars. Improving the robustness of vision algorithms is thus important to close the gap between human and machine perception and to enable safety-critical applications.
