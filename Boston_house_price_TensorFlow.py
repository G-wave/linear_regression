
# coding: utf-8

# In[27]:

import tensorflow as tf
import numpy as np
import sklearn
import pandas as pd
from sklearn.datasets import load_boston


# In[102]:

boston = load_boston()
bos = boston.data

alpha = 0.0001
num_epoch = 2000


# In[96]:

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
w = tf.Variable(np.random.randn(),name='Weight')
b = tf.Variable(np.random.randn(),name='Bias')


# In[97]:

# Linear Model
def model(X,w,b):
    return tf.add(tf.multiply(X,w),b)


# In[98]:

# Predictions and cost function
y_hat = model(X,w,b)
cost = tf.reduce_mean(tf.square(Y-y_hat))


# In[99]:

# Training step
train_op = tf.train.GradientDescentOptimizer(alpha).minimize(cost)


# In[103]:

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


# In[104]:

for epoch in range(num_epoch):
    for (x,y) in zip(bos[:,5],boston.target):
        sess.run(train_op, feed_dict = {X:x, Y:y})
    if epoch%100 == 0:
        c = sess.run(cost, feed_dict = {X:x, Y:y})
        print('epoch=',epoch,'cost=',c,'weight=',sess.run(w),'bias=',sess.run(b))


# In[ ]:




# In[ ]:



