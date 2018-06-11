import tensorflow as tf

print('starting')

a = tf.constant(3)
b = tf.constant(4)

c = a + b

with tf.Session() as sess:
    print('before file writer')
    File_Writer = tf.summary.FileWriter('../tf_logs', sess.graph)
    print(sess.run(c))
