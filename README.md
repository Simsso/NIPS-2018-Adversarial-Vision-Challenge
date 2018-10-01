# NIPS 2018 Adversarial Vision Challenge
This repository contains code, notes, and documents, related to the 2018 NIPS Adversarial Vision Challenge -- Pitting machine vision models against adversarial attacks.

## Links
* [Crowd.ai page](https://www.crowdai.org/challenges/nips-2018-adversarial-vision-challenge)
* [NIPS competition track](https://nips.cc/Conferences/2018/CompetitionTrack)

## Submitting Models

The root folder contains the necessary meta-files and references to submit a model to the challenge.

### The Model
[`run.sh`](./run.sh) first runs the [`setup.py`](./resnet-base/setup.py) script, which installs the 
[`SubmittableResNet`](./resnet-base/resnet_base/model/submittable_resnet.py). This is a simple wrapper around 
a `BaseModel` and provides a function that returns a foolbox model, representing the model itself.

### Submitting a New Model
If one wants to submit a newly implemented model (call it `MyModel`), the steps are as follows:

1. Train `MyModel` and create a `global_checkpoint`. Note the `CHECKPOINT_PATH` where the checkpoint can be found.
2. Change [`SubmittableResNet`](./resnet-base/resnet_base/model/submittable_resnet.py) to use `MyModel` as
its base class. 
3. If necessary, adjust [`submit.py`](./resnet-base/resnet_base/mains/submit.py) (e.g. change the placeholder
tensor if needed). In most cases however, this step should not be necessary.
4. Update [`run.sh`](./run.sh) and set the `CHECKPOINT_PATH` as the `global_checkpoint` argument
5. Head over to [crowdai's GitLab](https://gitlab.crowdai.org) and create a new repository. 
Add the new repository as a new remote of this repository, commit and push everything. 
6. Run `avc-submit .` (or - to test it locally - `avc-test-model .`) to run the evaluation and submit the model.  
