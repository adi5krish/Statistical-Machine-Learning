#!/usr/bin/env python
# coding: utf-8

# Because no work is required until a prediction is done, KNN is often referred to as lazy learning method

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
from tqdm import tqdm as tqdm


# In[2]:


def euclidean_measure(image1, image2):
    return np.sqrt(np.sum(np.subtract(image1,image2) ** 2))


# In[55]:


def predict(train_data, test_image, train_labels, k):
    distances = []
    for j in range(train_data.shape[0]):
        distances.append((euclidean_measure(train_data[0], test_image),j))
    distances.sort(key = lambda x : x[0])
#     return distances
    return predict_label(distances[:k],train_labels)

def predict_label(distances, train_labels):
    labels = list(train_labels[[item[1] for item in distances]])
    prediction = max(set(labels), key = labels.count)
    return prediction

def plot_curve(k, accuracies):
    plt.plot(k, accuracies)


# In[56]:


train_data, train_labels = loadlocal_mnist(images_path = 'F:\\Masters at ASU\\Semester 1\\SML\\train-images.idx3-ubyte',
                                            labels_path = 'F:\\Masters at ASU\\Semester 1\\SML\\train-labels.idx1-ubyte')
                                           
test_data, test_labels = loadlocal_mnist(images_path = 'F:\\Masters at ASU\\Semester 1\\SML\\t10k-images.idx3-ubyte',
                                            labels_path = 'F:\\Masters at ASU\\Semester 1\\SML\\t10k-labels.idx1-ubyte')


# In[57]:


test_data[0].shape


# In[58]:


K = [1, 3, 5, 10, 20, 30, 40, 50, 60]
accuracies = []
# print(predict(train_data, test_data[0], train_labels, k))
"""
print(t[:10])
d = train_labels[[item[1] for item in t]]
print(d.shape)

"""
for k in range(0,len(K)):
    count = 0
    for i in tqdm(range(test_data.shape[0])):
        predicted_label = predict(train_data, test_data[i], train_labels, K[k])
        actual_label = test_labels[i]
        if(predicted_label == actual_label):
            count = count+1
        
    print("Accuracy : {0}".format((count / test_data.shape[0]) * 100.0))
    accuracies.append((count / test_data.shape[0]) * 100.0)


# In[ ]:


plot_curve(k, accuracies)


# In[ ]:


a = [1,2,3]
b = [4,5,6]
print(np.sqrt(np.sum(np.subtract(a,b) ** 2)))


# In[ ]:


int(test_data.shape[0])


# In[48]:


np.zeros((2,2))


# In[ ]:




