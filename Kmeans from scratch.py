#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np


# In[2]:


df = pd.read_csv('F:\\Masters at ASU\\Semester 1\\SML\\CSE575-HW03-Data.csv')
df = np.array(df)
data = df


# In[3]:


def fix_missing_entries(data):
    if(data.shape[0] != 128):
        new_row = []
        
    for j in range(0,data.shape[1]):
        new_row.append(np.mean(data[j]))
    data = np.vstack([data,new_row])
    return data

data = fix_missing_entries(data)


# In[15]:


class kmeans:
    def __init__(self, k = 3, tolerance = 0.0001, max_iterations = 200):
        self.k = k
        self.tolerance  = tolerance
        self.max_iterations = max_iterations
        self.data = data
           
    def fit(self,data):
        self.centroids = {}

        #initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
        for i in range(self.k):
            self.centroids[i] = data[i]

        #begin iterations
        for i in range(self.max_iterations):
            self.classes = {}
            
            for i in range(self.k):
                self.classes[i] = []

            # find the distance between the point and cluster; choose the nearest centroid
            for features in data:
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)
        
            previous = dict(self.centroids)
    
            #average the cluster datapoints to re-calculate the centroids
            for label in self.classes:
                self.centroids[label] = np.average(self.classes[label], axis = 0)

            tol = 0.0
#             
            for j in range(self.k):
                                
                prv = previous[j]
                curr = self.centroids[j]
                tol = tol  + abs(np.sum([a-b for a,b in zip(prv, curr)]))
                                
            if(tol < self.tolerance):
                if(self.k == 2):
                    self.visualize(self.centroids, self.classes,self.k)
                print("converged", i)
                break
            else:
                continue
        return self.classes, self.centroids
        
    def predict(self, feature):
        
        distances = [np.linalg.norm(feature - self.centroids[i]) for i in self.centroids]
        label = distances.index(min(distances))
        return label
    
    def calculate_error(self, clusters, centroids):
        cummulative_error = 0.0
        for cluster in range(0,len(clusters)):
            cummulative_error = cummulative_error + np.sum(np.square(np.abs([np.subtract(point,centroids[cluster])                                                                 for point in clusters[cluster]])))
        return cummulative_error
            
  
    def visualize(self, final_centroids, classes, k):
        
        colors = { 0 : 'r' , 1 : 'b'}
        figure = plt.figure(figsize=(5, 5))
        
        for i in range(len(final_centroids)):  # let's visualize our centroids as x
            plt.scatter(final_centroids[i][0], final_centroids[i][1], s = 100, marker = 'x')
        
        for label in range(0,len(classes)):   # plot the points as per the class
            class_points = classes[label]
            
            for points in range(0,len(class_points)):
                plt.scatter(class_points[points][0], class_points[points][1], s = 30, color = colors[label])
        plt.show()
        figure.savefig('F:\\Masters at ASU\\Semester 1\\SML\\Assignment 3\\Kmeans\\Kmeans_clusters.jpg')


# In[16]:


K = np.arange(2,10)
errors = []

for j in range(0,len(K)):
    kmns = kmeans(k = K[j])
    clusters , centroids = kmns.fit(data)
    errors.append(kmns.calculate_error(clusters,centroids))
figure = plt.figure(figsize=(5, 5))
plt.plot(K, errors)
plt.show()
figure.savefig('F:\\Masters at ASU\\Semester 1\\SML\\Assignment 3\\Kmeans\\Kmeans_error_plot.jpg')

"""
for k in range(0,len(K)):
    if(K[k] == 2):
        kmns = kmeans(k = K[k])
        kmns.fit(data[:,:2])
    else:
        kmns.fit(data) 
# kmns.predict([])
"""


# In[ ]:


len(clusters)


# In[ ]:


data


# In[ ]:


np.sum(np.square(np.abs([np.subtract(point,centroids[0]) for point in clusters[0]])))


# In[ ]:




