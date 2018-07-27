import tensorflow as tf
sess = tf.Session()
hello = sess.run( tf.constant('hello world') )
print( hello )