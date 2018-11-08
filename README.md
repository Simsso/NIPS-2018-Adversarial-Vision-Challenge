# NIPS 2018 Adversarial Vision Challenge "Robust Model Track"
_[Timo I. Denk](https://timodenk.com/), [Florian Pfisterer](https://twitter.com/FlorianPfi), [Samed Guener](https://twitter.com/samedguener)  
(published in November 2018)_

**Abstract**. This repository contains code, documents, and deployment configuration files, related to our participation in the 2018 NIPS Adversarial Vision Challenge "Robust Model Track".  
We have implemented a technique called a LESCI layer which is supposed to increase the robustness of a neural network classifier.
It compresses the representation at a certain layer with a matrix computed using PCA on representations induced by correctly classified training samples at this same layer.
The compressed vector is being compared to an embedding space; and replaced with an embedding vector if a certain percentage of the _k_ most similar vectors belong to the same output label.  
In the current configuration, our method did not increase the robustness of the model as measured by the challenge, presumably because it comes with a decrease in classification accuracy.
We have formalized our approach in [this PDF](./docs/article/article.pdf).


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
* **Samed Güner** _(right; [@doktorgibson](https://github.com/doktorgibson))_: Training pipeline design, cloud administration, pipeline implementation, Go tool development.

## Repository

This repository is an integral component of our work and served the following purposes:
* **Code**. Version control our code base and keeping track of contributions other team members have made.
* **Project management tool**. We have used issues quite extensively to keep track of [work items](https://github.com/Simsso/NIPS-2018-Adversarial-Vision-Challenge/issues?q=label%3Awork-item+) and [meetings](https://github.com/Simsso/NIPS-2018-Adversarial-Vision-Challenge/issues?q=label%3Ameeting). For each meeting we took notes of assignments and documented the progress we had made.
* **Knowledge base**. The repository's [wiki](https://github.com/Simsso/NIPS-2018-Adversarial-Vision-Challenge/wiki) contains enduring information, such as [how our pipeline is set up](https://github.com/Simsso/NIPS-2018-Adversarial-Vision-Challenge/wiki/Training-Pipeline), which [papers we consider relevant](https://github.com/Simsso/NIPS-2018-Adversarial-Vision-Challenge/wiki/Reading-List), or [how we name our commits](https://github.com/Simsso/NIPS-2018-Adversarial-Vision-Challenge/wiki/Repository-Conventions), just to name a few.
* **Review**. Every contribution to the `master` branch had to go through a review process. In total we have have opened [more than 25 pull requests](https://github.com/Simsso/NIPS-2018-Adversarial-Vision-Challenge/pulls?q=is%3Apr); some of which have received more than 30 comments.

