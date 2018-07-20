import tensorflow as tf 
import data.test_data as test_data
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
        saver = tf.train.Saver()

    sess = tf.Session(graph=graph)
    with sess.as_default():
        # restore model
        print("Restoring model from %s" % (trainer.SAVER_PATH))
        saver.restore(sess, trainer.SAVER_PATH)
        print("Done restoring.")

        test_images = sess.run(images_batch)
        print("Got test images.")
        labels = sess.run([output_labels], feed_dict={images: test_images})
        print("Got labels.")

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
        if not i % 10:
            print("writing to file: %d/%d" % (i, len(labels)))
        result.append((filenames[i], class_ids[labels[i]]))

    with open(path, 'w+') as f:
        for res in result:
            f.write("%s %s\r\n" % (res))
