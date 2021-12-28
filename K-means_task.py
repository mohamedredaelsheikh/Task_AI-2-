#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


# Importing the dataset
dataset = pd.read_csv('Wuzzuf_Jobs.csv')


# In[3]:


#1.Factorize the YearsExp feature and convert it to numbers in new col
dataset['fact'] = pd.factorize(dataset['YearsExp'])[0]


# In[4]:


dataset.head()


# In[5]:


#2.Apply K-means for job title and companies.
dataset['fact1'] = pd.factorize(dataset['Title'])[0]
dataset['fact2'] = pd.factorize(dataset['Company'])[0]
X = dataset.iloc[:, [8, 9]].values
x=pd.DataFrame(X)


# In[6]:


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[7]:


plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[8]:


# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
y_kmeans == 0


# In[14]:


# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 15, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 15, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 15, c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('Clusters of jobs')
plt.xlabel('Titles')
plt.ylabel('Company')
plt.legend()
plt.show()


# In[ ]:




