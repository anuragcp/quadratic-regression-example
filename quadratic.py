import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import math

# Here we are going to find the h ad v intercept of (x-h)^2 + v

tf.get_default_graph()

y = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32)

h_est = tf.Variable(initial_value=0.0)
v_est = tf.Variable(initial_value=0.0)

y_est = tf.square(x-h_est)+v_est
cost = tf.pow((y_est-y),2)

train_out = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

##########################################################################################
#   Making a training dataset
h=1
v = -2

x_train = np.linspace(-2,4,201)
noise = np.random.randn(*x_train.shape) * 0.4
y_train = (x_train - h)**2 + v + noise

plt.rcParams['figure.figsize'] = (10,6)
plt.scatter(x_train, y_train)
plt.xlabel('x_train')
plt.ylabel('y_train')
out = plt.show(block = True)
time.sleep(5)
plt.close('all')
##########################################################################################
print("x_train Input : ", x_train)
print("y_train Output : ", y_train)
saver = tf.train.Saver()

init = tf.global_variables_initializer()

def train_model():
    with tf.Session() as sess:
        sess.run(init)
        for i in range(100):
            for x_input,y_input in zip(x_train, y_train):
                sess.run(train_out, feed_dict={x: x_input, y:y_input})

        h_true, v_true = sess.run([h_est,v_est])

    return [h_true,v_true]

h_true, v_true = train_model()

y_val = ((x_train-h_true)**2)+v_true

print("Horizontal_shift : {}   Vertical_shift : {}".format(h_true,v_true))
plt.scatter(x_train, y_train)
plt.plot(x_train ,y_val)
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.plot()
plt.show()
#out = plt.show(block = True)
time.sleep(5)
plt.close('all')



