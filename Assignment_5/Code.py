#!/usr/bin/env python
# coding: utf-8

# In[212]:


import numpy as np
from sklearn.cluster import DBSCAN
import sklearn
import matplotlib.pyplot as plt
import csv
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import pandas as pd 
from sklearn.preprocessing import StandardScaler

filename = "Patients.csv"
rows = [] 
  
with open(filename, 'r') as csvfile: 
    csvreader = csv.reader(csvfile) 
  
    for row in csvreader: 
        rows.append(list(map(int, row)))

data = np.array(rows)
data_nor= StandardScaler().fit_transform(data)
dimens = data.shape[1]

min_poin = 2*dimens
print("Min_points = " , min_poin)
neigh = NearestNeighbors(n_neighbors=min_poin-1)
nbrs = neigh.fit(data_nor)
distance, indices = nbrs.kneighbors(data_nor)


distance = np.sort(distance, axis=0)
distances = distance[:,1]
plt.plot(distances)
plt.show()
y = distances
x = range(1, len(y)+1)
kn = KneeLocator(x, y, curve='convex', direction='increasing')
elb = distances[kn.knee-1]

plt.xlabel('number of clusters k')
plt.ylabel('Distances(eligible for eplison)')
plt.plot(x, y)
plt.hlines(elb, plt.xlim()[0], plt.xlim()[1], linestyles='dashed')
plt.show()
print("Optimal value for epilson = ",elb)
model = DBSCAN(eps = elb , min_samples = min_poin).fit(data_nor)
print("Labels = ",model.labels_)
al = sum(model.labels_== -1)
print("total number of outiners = ",al)
lab = model.labels_
new_lbs = abs(lab)
pd.DataFrame(new_lbs).to_csv("output.csv", header=None, index=None)

