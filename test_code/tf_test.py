import tensorflow as tf

# var = tf.Variable(5)
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     print(sess.run(var))

graph = tf.Graph()

with graph.as_default():
    var = tf.Variable(5)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(var.eval())