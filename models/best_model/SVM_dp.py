
# coding: utf-8

# In[13]:


import os
import sys
import os
import dlib
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ChangedBehaviorWarning


# In[3]:


def list_all_files(rootdir):
    _files = []
    list = os.listdir(rootdir)
    for i in range(0,len(list)):
        path = os.path.join(rootdir,list[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
            _files.append(path)
    return _files


# In[7]:


files1 = list_all_files('depressed/')
files2 = list_all_files('nondepressed/')
lable1 = [0] * len(files1)
lable2 = [1] * len(files2)


# In[6]:


predictor_path = './shape_predictor_68_face_landmarks.dat'
face_rec_model_path = './dlib_face_recognition_resnet_model_v1.dat'
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)


# In[10]:


X = []
Y = []
for i,j in zip(files1+files2,lable1+lable2):
    image = cv2.imread(i)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)
    for k, d in enumerate(dets):
        shape = sp(image, d)
        face_descriptor = facerec.compute_face_descriptor(image, shape)
        v = np.array(face_descriptor)
        X.append(v)
        Y.append(j)
        print(len(X),end=' ')
        break
np.save('X.npy',X)
np.save('Y.npy',Y)


# In[11]:


X = np.load('X.npy')
Y = np.load('Y.npy')
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.25)


# In[26]:
clf = svm.SVC(C=55,kernel='linear',gamma=0.005)
clf.fit(x_train, y_train)


# In[27]:


print (clf.score(x_train, y_train)) 
print ('Training accuracy：', accuracy_score(y_train, clf.predict(x_train)))
print (clf.score(x_test, y_test))
print ('Testing accuracy：', accuracy_score(y_test, clf.predict(x_test)))
print ('decision_function:\n', clf.decision_function(x_train))
print ('\npredict:\n', clf.predict(x_train))

