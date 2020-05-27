#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np


# In[11]:


df = pd.read_csv('F:\\Masters at ASU\\Semester 1\\SML\\CSE575-HW03-Data.csv')
df = np.array(df)
data = df


# In[12]:


def fix_missing_entries(data):
    if(data.shape[0] != 128):
        new_row = []
        
    for j in range(0,data.shape[1]):
        new_row.append(np.mean(data[j]))
    data = np.vstack([data,new_row])
    return data

data = fix_missing_entries(data)


# In[20]:


class kmeans:
    def __init__(self, k = 3, tolerance = 0.01, max_iterations = 200):
        self.k = k
        self.tolerance  = tolerance
        self.max_iterations = max_iterations
        self.data = data
           
    def fit(self,data):
        self.centroids = {}

        #initialize the centroids
        for i in range(self.k):
            self.centroids[i] = np.random.random_sample((1,13)) 
            
        self.classes = {}
    
        for clss in range(self.k):
            self.classes[clss] = []
            
        #begin iterations
        for iteration in range(self.max_iterations):
        
            tol = 0.0                
            # find the distance between the point and cluster; choose the nearest centroid
            for feature in data:

                distances = [np.linalg.norm(feature - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(feature)
            
            previous = dict(self.centroids)
            epsilon = 0.01
            

            # re-calculate the centroids 
            for label in self.classes:
                numerator = 0.0
                denominator = 0.0
                converged = False
                
                for point in self.classes[label]:
                    
                    diff = (np.sum((point - self.centroids[label])** 2)**0.5)
                    
                    if(diff == 0):
                        return self.classes, previous
                    numerator+= (point / diff)
                    denominator+= (1 / diff)
                    
                if(denominator != 0):
                    self.centroids[label] = previous[label]
                    
            current = dict(self.centroids)
            
            for j in range(self.k):
                prv = previous[j]
                curr = current[j]
        
                tol = tol + abs(np.sum([a-b for a,b in zip(prv, curr)]))
            
            if(tol < self.tolerance):
                print("converged in iteration no ", iteration)
                break

        print(self.centroids)
        return self.classes, self.centroids
             

    def calculate_error(self, clusters, centroids):
        cummulative_error = 0.0
        for cluster in range(0,len(clusters)):
            cummulative_error = cummulative_error + np.sum(np.square(np.abs([np.subtract(point,centroids[cluster])                                                                 for point in clusters[cluster]])))
        return cummulative_error
    
    def visualize(self, final_centroids, classes):
    
        colors = { 0 : 'r' , 1 : 'b'}

        for i in range(0,len(final_centroids)):  # let's visualize our centroids as x
            plt.scatter(final_centroids[i][0], final_centroids[i][1], s = 100, marker = 'x')
        
        for label in range(0,len(classes)):   # plot the points as per the class
            class_points = classes[label]
            
            for points in range(0,len(class_points)):
                plt.scatter(class_points[points][0], class_points[points][1], s = 30, color = colors[label])
    """
    def predict(self, feature)
        
        distances = [np.linalg.norm(feature - self.centroids[i]) for i in self.centroids]
        label = distances.index(min(distances))
        return label   
"""


# In[21]:


np.random.seed(42)
K = np.arange(2,10)
errors = []

for j in range(0,len(K)):
#         data = data[:,:2]
    kmns = kmeans(k = K[j])
    clusters , centroids = kmns.fit(data)
#     if(K[j] == 2):
#         kmns.visualize(centroids, clusters)
    """
    if(K[j] == 2):
        kmns.visualize(centroids,clusters)
    """
    errors.append(kmns.calculate_error(clusters,centroids))

figure = plt.figure(figsize = (5,5))
plt.plot(K, errors)
plt.show()
figure.savefig('F:\\Masters at ASU\\Semester 1\\SML\\Assignment 3\\Kmeans Alternative\\KmeansAlternative.jpg')


# In[ ]:


len(clusters)


# In[ ]:


data


# In[ ]:


np.sum(np.square(np.abs([np.subtract(point,centroids[0]) for point in clusters[0]])))


# In[ ]:


import numpy as np
np.random.choice(range(20), 10, replace = False)


# In[ ]:


for feature in data:
    print(feature)


# In[ ]:


import numpy as np
np.random.random_sample((1,13))


# In[ ]:


a = np.array([1,2,3])
b = np.array([2,3,4])
np.linalg.norm(np.subtract(a,b))


# In[ ]:


np.linalg.norm(np.sum((a-b)))


# In[ ]:


(np.sum((a-b)**2))**0.5


# In[ ]:




