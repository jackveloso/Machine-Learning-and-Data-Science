import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.test.labels[5])
plt.imshow(mnist.test.images[5].reshape((28, 28)), cmap="gray")
plt.show()
# modelling of the computational graph/neural network
x = tf.placeholder(tf.float32, [None, 784], name="data")
# first layer
with tf.name_scope("layer1"):
    W1 = tf.Variable(tf.truncated_normal((784, 30), stddev=2/np.sqrt(784)),
                     name="W1")
    b1 = tf.Variable(tf.zeros([30]), name="b1")
    hidden1 = tf.matmul(x, W1)+b1
with tf.name_scope("ReLU1"):
    hidden1ReLU = tf.nn.relu(hidden1)
# second layer
with tf.name_scope("layer2"):
    W2 = tf.Variable(tf.truncated_normal((30, 30), stddev=2/np.sqrt(30)),
                     name="W2")
    b2 = tf.Variable(tf.zeros([30]), name="b2")
    hidden2 = tf.matmul(hidden1ReLU, W2)+b2
with tf.name_scope("ReLU2"):
    hidden2ReLU = tf.nn.relu(hidden2)
# third layer
with tf.name_scope("output_layerrr"):
    W3 = tf.Variable(tf.truncated_normal((30, 10), stddev=2/np.sqrt(30)),
                     name="W3")
    b3 = tf.Variable(tf.zeros([10]), name="b3")
    Phi = tf.matmul(hidden2ReLU, W3)+b3
# placeholder for the labels
z = tf.placeholder(tf.float32, [None, 10])
# defining the loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=z,
                                                                 logits=Phi))
# defining which optimizer we use with what learningrate
learningrate = 0.1
train_step = tf.train.GradientDescentOptimizer(learningrate).minimize(loss)
sess = tf.InteractiveSession()
# initialzing the weights and biases
tf.global_variables_initializer().run()
for _ in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, z: batch_ys})
correct_prediction = tf.equal(tf.argmax(Phi, 1), tf.argmax(z, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    z: mnist.test.labels}))
plt.imshow(mnist.test.images[10].reshape((28, 28)), cmap="gray")
plt.show()
Z = Phi.eval(feed_dict={x: mnist.test.images[10].reshape(1, 784)})
print(sess.run(tf.nn.softmax(Z)))
sess.close()
