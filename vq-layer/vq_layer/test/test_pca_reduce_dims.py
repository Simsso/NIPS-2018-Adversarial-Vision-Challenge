import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from vq_layer.test.tf_test_case import TFTestCase
from vq_layer.vq_layer import pca_reduce_dims


class TestPCAReduceDims(TFTestCase):
    """
    Test the PCA dimensionality reduction function.
    """

    def setUp(self) -> None:
        super(TestPCAReduceDims, self).setUp()
        np.random.seed(42)

    def feed(self, x_val: np.ndarray, num_components: int):
        x = tf.constant(x_val, dtype=tf.float32, shape=x_val.shape)
        reduced_result = pca_reduce_dims(x, num_components)
        # self.init_vars()

        result = self.sess.run(reduced_result)

        pca = PCA(num_components, svd_solver='full')
        correct_projection = pca.fit_transform(x_val)

        # TODO problem: the sign of the eigenvalues can be chosen arbitrarily, so how to test this correctly (need to
        # get the signs of the eigenvectors that the TF-function used!)?
        # using abs is non entirely correct, because once the eigenvalues have the same sign, the values must match
        # exactly, including their signs

        # needs larger tolerance than usual
        self.assert_numerically_equal(np.abs(result), np.abs(correct_projection), rtol=1e-2, atol=1e-3)

    def test_pca_1(self):
        x_val = np.random.uniform(low=-1, high=1, size=[5, 3])
        self.feed(x_val, num_components=2)

    def test_pca_2(self):
        x_val = np.random.uniform(low=-10, high=-5, size=[10, 100])
        self.feed(x_val, num_components=10)

    def test_pca_3(self):
        x_val = np.array([[2, 4, 6], [1, 2, 3], [10, 20, 30], [100, 200, 300]], dtype=np.float32)
        self.feed(x_val, num_components=1)
