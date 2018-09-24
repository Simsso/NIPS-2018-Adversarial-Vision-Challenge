# ResNet Base
_This folder contains a self-contained ResNet model which provides the basis for our future experiments._

It's core is a transferred
[official ResNet model](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/resnet_v2.py)
incorporated into a popular [TensorFlow project template](https://github.com/MrGemy95/Tensorflow-Project-Template)
(which we have adapted to our needs). 

### Current Functionality
As of now, the following features are implemented:
* restore the pre-trained [ALP weights](https://github.com/tensorflow/models/tree/master/research/adversarial_logit_pairing)
* load the [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) training and validation set
* channel training and validation data via a `tf.data` input pipeline
* evaluate the model on the validation set (using [`eval.py`](resnet_base/mains/eval.py))
* support multiple savers to restore and save different parts of the graph, namely pre-trained, custom and global weights
* perform training using [`resnet_trainer.py`](resnet_base/trainer/resnet_trainer.py)
(can be tested with [`test_train.py`](resnet_base/mains/test_train.py))

The current code is still not production-ready. Refactoring and documentation are needed. 


### Local Execution
Install with 
```bash
python setup.py install
```
and run using
```bash
python -m resnet_base
```
