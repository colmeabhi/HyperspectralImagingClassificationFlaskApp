#!/usr/bin/env python
# coding: utf-8

# In[41]:


# imported libraries

from scipy.io import loadmat
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


# In[23]:


# Main code

dataset_file = loadmat('C:\\Users\\ahirr\\Desktop\\Indian_pines_corrected.mat') 
gt_file = loadmat('C:\\Users\\ahirr\\Desktop\\Indian_pines_gt.mat')

dataset = dataset_file[ 'indian_pines_corrected']
gt = gt_file['indian_pines_gt']

X = np.reshape(dataset, (21025,200)) # your way gives good accuracy
y = gt.reshape(145*145,1)

# Normalisation of data

normalized_X =  preprocessing.normalize(X)

# Remove the rows with 0 gt values

zero_results_indexes = []
for i in range(len(normalized_y)):
    if(y[i] == 0):
        zero_results_indexes.append(i)
        
y_del_zero, X_del_zero = np.delete(y, zero_results_indexes), np.delete(normalized_X, zero_results_indexes, axis = 0)

print(len(X_del_zero))
print(len(y_del_zero))


# In[25]:


# Train,Test Splitting of data

from sklearn.model_selection import train_test_split

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X_del_zero, y_del_zero, test_size=0.3, random_state=3)


# In[26]:


# Prediction Using decision tree Algo 

Clf_dt = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

print(Clf_dt) # it shows the default parameters

Clf_dt.fit(X_trainset,y_trainset)
predTree = Clf_dt.predict(X_testset)


# In[27]:


# Metrics and Accuracy

from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


# In[49]:


scalar = StandardScaler()
X_train = scalar.fit_transform(X_trainset)
X_test = scalar.transform(X_testset)

from sklearn.neural_network import MLPClassifier
clf_mlp = MLPClassifier(hidden_layer_sizes=(100,150,100), max_iter = 300, activation = 'identity', learning_rate = 'constant').fit(X_train, y_trainset)
clf_mlp.predict_proba(X_testset[:1])


# In[48]:


# Accuracy of the Mlp Classification

print('accuracy with MLP:'+str(clf_mlp.score(X_test, y_testset)))


# In[ ]:




