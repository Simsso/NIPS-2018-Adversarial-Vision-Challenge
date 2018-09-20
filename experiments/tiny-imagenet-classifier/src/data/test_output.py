import tiny_imagenet as data 
import glob
import tensorflow as tf

def load_test_image_filenames():
    filenames = glob.glob(data.PATH + '/test/images/*.JPEG')
    return filenames


def test_batch():
    filenames = load_test_image_filenames()
    filename_q = tf.train.input_producer(filenames, shuffle=False)

    # 2 read_image threads to keep batch_join queue full:
    result = tf.train.batch_join([data.read_image(filename_q, mode='test')],
                                 batch_size=len(filenames), shapes=[(56, 56, 3), ()], capacity=2048)
    return result


