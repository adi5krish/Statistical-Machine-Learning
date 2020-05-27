#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
import matplotlib.pyplot as plt


# In[12]:


class GMM:
    def __init__(self, k, tolerance = 0.01):
        
        self.k = k
        self.tolerance = tolerance
        self.colors = ['red','blue']
        
    def init(self, X):        # X is the full data
        
        self.shape = X.shape
        self.n, self.m = self.shape
        
        self.phi = np.full(shape = self.k, fill_value = 1 / self.k)
        self.weights = np.full(shape = self.shape, fill_value = 1 / self.k)
        
        random_row = np.random.randint(low = 0, high = self.n, size = self.k)
        self.mu = [ X[row_index, :] for row_index in random_row ]
        self.sigma  = [ np.cov(X.T) for j in range(self.k)]
        
    def expectation(self, X):
        
        self.weights = self.predict_probability(X)
        self.phi = self.weights.mean(axis = 0)
        
    def maximization(self, X):
        
        for i in range(self.k):
            
            weight = self.weights[: , [i]]
            total_weight = weight.sum()
            self.mu[i] = (X * weight).sum(axis = 0) / total_weight
            self.sigma[i] = np.cov(X.T, aweights =  (weight/total_weight).flatten(), bias = True)
    
    def fit(self, X):
        
        self.init(X)
        
        prev_mu = self.mu
    
        updated_mu = np.zeros((self.k, self.m))
        
        while(abs(np.sum([a-b for a,b in zip(prev_mu,updated_mu)])) > self.tolerance):
            
            prev_mu = self.mu
            self.expectation(X)
            self.maximization(X)
            updated_mu = self.mu
            
            if(abs(np.sum([a-b for a,b in zip(prev_mu,updated_mu)])) < self.tolerance):
                print("Algorithm converged")
                self.visualize(X, self.mu, self.sigma, self.colors)
                break
        
    def predict_probability(self, X):
        
        likelihood = np.zeros((self.n , self.k))
        
        for i in range(self.k):
            distribution = multivariate_normal(mean = self.mu[i], cov = self.sigma[i])
            likelihood[:,i] = distribution.pdf(X)
        
        numerator = likelihood * self.phi
        denominator = numerator.sum(axis = 1)[:, np.newaxis]
        weights = numerator / denominator
        return weights
    
    def predict(self, X):
        
        weights = self.predict_probability(X)
        return np.argmax(weights, axis = 1)
    
    def visualize(self, data, cluster_means, cov, colors):
        likelihood = []
        classes = {}
        figure = plt.figure(figsize = (5,5))
        
        for c in range(self.k):
            classes[c] = []

        for c in range(self.k):
            likelihood.append(multivariate_normal.pdf(x = data, mean = cluster_means[c], cov = cov[c]))
        likelihood = np.array(likelihood)
        predictions = np.argmax(likelihood, axis = 0)

        for i in range(0,len(data)):
            label = predictions[i]
            classes[label].append(data[i])

        for i in range(len(cluster_means)):  
            plt.scatter(cluster_means[i][0], cluster_means[i][1], s = 100, marker = 'x')

        for label in range(0,len(classes)):   # plot the points as per the class
            class_points = classes[label]

            for points in range(0,len(class_points)):
                plt.scatter(class_points[points][0], class_points[points][1], s = 30, color = colors[label])
        plt.show()
        figure.savefig('F:\\Masters at ASU\\Semester 1\\SML\\GMM.jpg')


# In[13]:


df = pd.read_csv('F:\\Masters at ASU\\Semester 1\\SML\\CSE575-HW03-Data.csv')
df = np.array(df)
data = df


# In[14]:


def fix_missing_entries(data):
    if(data.shape[0] != 128):
        new_row = []
        
    for j in range(0,data.shape[1]):
        new_row.append(np.mean(data[j]))
    data = np.vstack([data,new_row])
    return data

data = fix_missing_entries(data)


# In[15]:


# fit a model
np.random.seed(42)
gmm = GMM(k = 2)
gmm.fit(data[:,:2])


# In[6]:


updated_mu = np.zeros((2,2))
prev_mu = np.random.rand(2,2)
np.sum([a-b for a,b in zip(prev_mu,updated_mu)])


# In[7]:


prev_mu


# In[ ]:




