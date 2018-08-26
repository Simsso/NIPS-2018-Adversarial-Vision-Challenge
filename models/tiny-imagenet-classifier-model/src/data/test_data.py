import data.tiny_imagenet as data 
import glob
import tensorflow as tf

def load_test_image_filenames():
    filenames = glob.glob(data.PATH + '/test/images/*.JPEG')
    return filenames


def read_single_test_image(filename_q):
    filename = filename_q.dequeue()
    f = tf.read_file(filename)
    img = tf.image.decode_jpeg(f, channels=3)
    img = tf.image.crop_to_bounding_box(img, 4, 4, 56, 56)

    return [img]


def test_batch():
    filenames = load_test_image_filenames()
    filename_q = tf.train.input_producer(filenames, shuffle=False)

    # 2 read_image threads to keep batch_join queue full:
    result = tf.train.batch_join([read_single_test_image(filename_q)],
                                 batch_size=len(filenames), shapes=[(56, 56, 3)], capacity=2048)
    return result


