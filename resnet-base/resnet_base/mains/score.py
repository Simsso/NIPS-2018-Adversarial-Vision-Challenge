import tensorflow as tf
import os
import numpy as np
from PIL import Image

from foolbox.criteria import Misclassification
import foolbox.attacks

from resnet_base.data.tiny_imagenet_pipeline import TinyImageNetPipeline
from resnet_base.model.submittable_resnet import SubmittableResNet

VALIDATION_IMAGES_DIR = os.path.expanduser('~/.data/tiny-imagenet-200/val/images')
NUM_IMAGES_PER_ATTACK = 1

__attacks = [
    foolbox.attacks.FGSM,
    foolbox.attacks.GradientSignAttack,
    foolbox.attacks.IterativeGradientSignAttack,
    foolbox.attacks.IterativeGradientAttack,
    foolbox.attacks.LBFGSAttack,
    foolbox.attacks.ApproximateLBFGSAttack,
    foolbox.attacks.DeepFoolAttack,
    foolbox.attacks.DeepFoolL2Attack,
    foolbox.attacks.DeepFoolLinfinityAttack,
    foolbox.attacks.SaliencyMapAttack,
    foolbox.attacks.GaussianBlurAttack,
    foolbox.attacks.ContrastReductionAttack,
    foolbox.attacks.SinglePixelAttack,
    foolbox.attacks.LocalSearchAttack,
    foolbox.attacks.SLSQPAttack,
    foolbox.attacks.AdditiveNoiseAttack,
    foolbox.attacks.AdditiveUniformNoiseAttack,
    foolbox.attacks.AdditiveGaussianNoiseAttack,
    foolbox.attacks.BlendedUniformNoiseAttack,
    foolbox.attacks.SaltAndPepperNoiseAttack,
    foolbox.attacks.PrecomputedImagesAttack,
    foolbox.attacks.BoundaryAttack
]


def main(args):
    """
    Runs a number of foolbox attacks on a SubmittableResNet. Used to evaluate the robustness of the model.
    Prints the median and mean L2 pixel distances between clean and adversarial images for all successful attacks.
    """
    w = TinyImageNetPipeline.img_width
    h = TinyImageNetPipeline.img_height
    c = TinyImageNetPipeline.img_channels

    sess = tf.Session()
    with sess.as_default():
        images = tf.placeholder(tf.float32, (None, w, h, c), name="images")
        model = SubmittableResNet(x=images)
        foolbox_model = model.get_foolbox_model()

    # specify what kind of attack we will do
    criterion = Misclassification()

    l2_distances = []
    for Attack in __attacks:
        # perform attacks on a number of random validation images
        for _ in range(NUM_IMAGES_PER_ATTACK):
            attack = Attack(foolbox_model, criterion)

            image = __get_random_image()
            label = np.argmax(foolbox_model.predictions(image))

            adversarial_image = attack(image, label=label)

            if adversarial_image is not None:
                # compare pixels to original image
                l2_pixel_distance = np.sum(np.square((image - adversarial_image) / 255.))
                l2_distances.append(l2_pixel_distance)
                print("{} *did* produce an attack!".format(attack.name()))
            else:
                print("{} could not find an adversarial image.".format(attack.name()))

    # calculate the mean and median l2 distances
    num_attacks = len(l2_distances)
    l2_distances = np.array(l2_distances)
    mean = np.mean(l2_distances)
    median = np.median(l2_distances)

    print("Evaluation done on {} successful attacks.\n\tMean L2 distance: {}\n\tMedian L2 distance: {}"
          .format(num_attacks, mean, median))


def __get_random_image() -> np.ndarray:
    """
    Chooses a random image from the Tiny ImageNet validation set and returns it's np.ndarray representation.
    """
    image = None

    # some images appear to have a different shape => make sure we get a 'good' one with shape (64, 64, 3)
    while image is None or image.shape != (64, 64, 3):
        # get a random validation image index
        random_image_index = np.random.randint(low=0, high=TinyImageNetPipeline.num_valid_samples, size=1)[0]
        path = os.path.join(VALIDATION_IMAGES_DIR, 'val_{}.JPEG'.format(random_image_index))
        image = np.asarray(Image.open(path), dtype=np.float32)

    return image


if __name__ == '__main__':
    tf.app.run()
