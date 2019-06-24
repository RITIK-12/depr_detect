
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# In[10]:


emotion_data = np.load('emotion_data.npy')
Y = np.load('Y.npy')


# In[11]:


y_ = []
for i in Y:
    y = np.zeros(2,dtype=np.float)
    y[i] = 1.0
    y_.append(y)


# In[12]:


x_train, x_test, y_train, y_test = train_test_split(emotion_data, y_, random_state=0, test_size=50)


# In[14]:


keep_prob = tf.placeholder(tf.float32,name='keepprob')
X = tf.placeholder(tf.float32, [None, 4096], name="X")
Y = tf.placeholder(tf.uint8, [None,2], name="Y")
w1 = tf.Variable(tf.random_normal([4096, 1024], stddev=0.01),name='w1')
b1 = tf.Variable(tf.random_normal([1024], stddev=0.01))
w2 = tf.Variable(tf.random_normal([1024, 256], stddev=0.01))
b2 = tf.Variable(tf.random_normal([256], stddev=0.01))
w3 = tf.Variable(tf.random_normal([256, 2], stddev=0.01))
b3 = tf.Variable(tf.random_normal([2], stddev=0.01))


# In[15]:


fc1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(X, w1), b1)),keep_prob)
fc2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(fc1, w2), b2)),keep_prob)
fc3 = tf.add(tf.matmul(fc2, w3), b3, name='fc3')
out = tf.nn.softmax(fc3,name='out')
reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=fc3))+reg
opt = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc3, 1),tf.argmax(Y,1)),tf.float32))


# In[ ]:


saver=tf.train.Saver(max_to_keep=1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for j in range(400):
        los,acc,_ = sess.run([loss,accuracy,opt],feed_dict={X:x_train,Y:y_train,keep_prob:0.5})
        print('step:{}    loss:{}    acc:{}'.format(j,los,acc))
        if j % 10 == 0 and j != 0:
            acc = sess.run(accuracy,feed_dict={X:x_test,Y:y_test,keep_prob:1.0})
            print('Test accuracy：',acc)
        if j % 100 == 0 and j != 0:
            saver.save(sess, './model/m')
    saver.save(sess, './model/m')

