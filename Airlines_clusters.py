# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 19:36:08 2020

@author: Smita Gavandi
"""

import pandas as pd
import matplotlib.pylab as plt
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np

filename="D:\\R Excel Sessions\\Assignments\\Clustering\\EastWestAirlines.xlsx"

Airlines = pd.read_excel(filename,sheet_name='data')

Airlines.head()
Airlines.shape
Airlines.columns
Airlines.dtypes

#Removing id column
Airlines2 = Airlines.iloc[:,1:]
Airlines2.columns

#####Visualizations#####################################################3

# Box and Whisker Plots
from matplotlib import pyplot

Airlines2.plot(kind='box', subplots=True, layout=(4,3), sharex=False, sharey=False)
pyplot.show()

# Pairwise Pearson correlations

correlations = Airlines2.corr(method='pearson')
print(correlations)

# Scatterplot Matrix
from matplotlib import pyplot

from pandas.plotting import scatter_matrix

scatter_matrix(Airlines)
pyplot.show
pyplot.savefig('D:\\R Excel Sessions\\Assignments\\Clustering\\scatterplot.png')

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Airlines2.iloc[:,:])
df_norm.head()

###### screw plot or elbow curve ############

k=list(range(2,23))

k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 

model=KMeans(n_clusters=5)
model.fit(df_norm)
model.labels_
md=pd.Series(model.labels_)
Airlines['clust']=md
df_norm.head()
Airlines.head()

Grouped_means=Airlines.iloc[:,1:].groupby(Airlines.clust).mean()

# creating a csv file 
Airlines.to_csv("Airline_with clust.csv",encoding="utf-8")

import os

os.getcwd()
os.chdir('D:\\R Excel Sessions\\Assignments\\Clustering')


#################################################################################

import pandas as pd
import matplotlib.pylab as plt
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np

crime = pd.read_csv("D:\\R Excel Sessions\\Assignments\\Clustering\\crime_data.csv")

crime.head
crime.columns
crime.dtypes
crime.shape
crime.info()

####To check missing values in the data
print("******Missing values in the dataset*****")
print(crime.isna().sum())
print("\n")

# Fill missing values with mean column values in the train set
#train.fillna(train.mean(), inplace=True)



####Excluding Country column
crime2=crime.iloc[:,1:]
crime2.columns

# Scatterplot Matrix
from matplotlib import pyplot

from pandas.plotting import scatter_matrix

scatter_matrix(crime2)
pyplot.show
pyplot.savefig('D:\\R Excel Sessions\\Assignments\\Clustering\\crime_scatterplott.png')

# Density Plots
from matplotlib import pyplot

crime2.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
pyplot.show()

#####Correlation plot as heatmap
import seaborn as sns
crime2.columns
cor = crime2.corr() #Calculate the correlation of the above variables
sns.heatmap(cor, square = True) #Plot the correlation as heat map

# Univariate Histograms
from matplotlib import pyplot

crime2.hist()
pyplot.show()

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crime2)
df_norm.head()

###### screw plot or elbow curve ############

k=list(range(2,23))

k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 

model=KMeans(n_clusters=5)
y_kmeans=model.fit(df_norm)
model.labels_
md=pd.Series(model.labels_)
crime2['clust']=md
df_norm.head()
crime2.head()
crime2['clust'].size

Grouped_means=crime2.groupby(crime2.clust).mean()

#####Size for each cluster
#Store the labels
labels = model.labels_

#Then get the frequency count of the non-negative labels
counts = np.bincount(labels[labels>=0])

print(counts)

# creating a csv file 
crime2.to_csv("crime2_with clust1.csv",encoding="utf-8")

import os

os.getcwd()
os.chdir('D:\\R Excel Sessions\\Assignments\\Clustering')



plt.scatter(crime2.iloc[:,0],crime2.iloc[:,1],crime2.iloc[:,2],crime2.iloc[:,3],c=model.labels_,cmap='brg')
crime2.iloc[:,0]
