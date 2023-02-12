#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster 
import pandas as pd
from scipy.cluster.vq import kmeans, vq 
from scipy.cluster.vq import whiten
import os
import numpy as np
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time


#import random
#random.seed((1000, 2000))


# In[ ]:


x_coordinates = [80.1, 93.1, 86.6, 98.5, 86.4, 9.5, 15.2, 3.4, 10.4, 20.3, 44.2, 56.8, 49.2, 62.5, 44.0]
y_coordinates = [87.2, 96.1, 95.6, 92.4, 92.4, 57.7, 49.4, 47.3, 59.1, 55.5, 25.6, 2.1, 10.9, 24.1, 10.3]

plt.scatter(x_coordinates, y_coordinates)
plt.show()
#df is used to store points
df = pd.DataFrame({'x_coordinate': x_coordinates, 'y_coordinate': y_coordinates})


# In[ ]:


#computes the distance between all pairs of clusters
z = linkage(df, 'ward')
#fcluster generates clusters and assigns associated cluster labels
# to a new column in the DataFrame.
df['cluster_labels'] = fcluster(z, 3, criterion = 'maxclust')


# In[4]:


sns.scatterplot(x = 'x_coordinate', y = 'y_coordinate', hue = 'cluster_labels', data = df)
plt.show()


# In[5]:


centroids,_ = kmeans(df, 3)
df['cluster_labels'],_ = vq(df, centroids)


# In[6]:


sns.scatterplot(x = 'x_coordinate', y = 'y_coordinate', hue = 'cluster_labels', data = df)
plt.show()


# In[7]:


#What is normalization of data? It is a process by
#which we rescale the values of a variable
#with respect to standard deviation of the data.
data = [5, 1, 3, 3, 2, 3, 3, 8, 1, 2, 2, 3, 5]


# In[8]:


#whiten divides each element by the standard deviation of the column
scaled_data = whiten(data)
print(scaled_data)


# In[9]:


plt.figure(figsize=(10,5))
plt.plot(data, label = 'original', color = "blue")
plt.plot(scaled_data, label = 'scaled', color = "red")
plt.legend()
plt.xlabel('position')
plt.ylabel('values') 
plt.show()
#y-axis is for the values, x-axis is for the position


