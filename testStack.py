import tensorflow as tf

a = tf.constant(2,tf.int32,[4,5,3])
b = tf.unstack(a,axis= 2)
c = []
for i in range(3):
    c.append(tf.reshape(b[i],[1,-1]))
d = tf.concat(c,axis = 1)
with tf.Session() as sess:
    print(d.shape)
    pass