import tensorflow as tf 
import data.test_output as test_data
import data.tiny_imagenet as data
import model.simple_resnet as cnn
import trainer.adam_resnet as trainer

def test_inference():
    """ Runs the simple_resnet model on the test images and returns the argmax labels.
    """
    graph = tf.Graph()
    with graph.as_default():
        # input
        images = tf.placeholder(tf.float32, shape=[None, data.IMG_DIM, data.IMG_DIM, data.IMG_CHANNELS], name='images')

        # data set
        images_batch = test_data.test_batch()
        _, softmax = cnn.model(tf.cast(images, tf.float32))
        output_labels = tf.argmax(softmax, axis=1)

    sess = tf.Session(graph=graph)
    saver = tf.train.Saver()
    with sess.as_default():
        # restore model
        saver.restore(sess, trainer.SAVER_PATH)

        test_images = sess.run(images_batch)
        labels = sess.run([output_labels], feed_dict={images: test_images})

    return labels

def write_test_results(path):
    labels = test_inference()

    # we need the class-id, not the index
    label_dict, _ = data.build_label_dicts()
    class_ids = {c_index : c_label for c_label, c_index in label_dict.items()}

    filenames = test_data.load_test_images()
    result = []

    assert len(filenames) == len(labels)

    for i in range(len(labels)):
        result.append((filenames[i], class_ids[labels[i]]))

    with open(path, 'w+') as f:
        for res in result:
            f.write("%s %s\r\n" % (res))
