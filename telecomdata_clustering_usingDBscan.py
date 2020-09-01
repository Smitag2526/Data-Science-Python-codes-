# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 21:41:44 2020

@author: Smita Gavandi
"""

# Data handling
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn import preprocessing
from sklearn.metrics import silhouette_score

# Dimensionality reduction
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Visualization
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

%matplotlib inline 

pd.set_option('display.max_columns', 500)

df= pd.read_csv(r"C:\Users\Smita Gavandi\Documents\python_excelR\LENOVO\R Excel Sessions\Assignments\Clustering\CustomerSegmentation-master\data.csv")
df.head()
df.info()
df.describe()
df.shape
 
df.isnull().sum()

df=df.loc[df.TotalCharges!=" ",:]
#df.TotalCharges = df.TotalCharges.astype(float) 


replace_cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport','StreamingTV', 'StreamingMovies', 'Partner', 'Dependents',
                   'PhoneService', 'MultipleLines', 'PaperlessBilling', 'Churn']
    
for i in replace_cols:
       df.loc[:,i] = df.loc[:,i].replace({'No internet service' : 'No','No phone service':'No'})
       df.loc[:,i] = df.loc[:,i].map({'No':0,'Yes':1})
df.gender=df.gender.map({"Female":0,"Male":1})

df.head()

others_categorical = ['Contract', 'PaymentMethod', 'InternetService']
for i in others_categorical:
    df = df.join(pd.get_dummies(df[i], prefix=i))
df.drop(others_categorical, axis=1, inplace=True)

df.info()


# Calculate number of services
services = ['PhoneService', 'MultipleLines', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'InternetService_DSL', 'InternetService_Fiber optic',
            'InternetService_No']
df['nr_services'] = df.apply(lambda row: sum([row[x] for x in services[:-1]]), 1)
df.info()    
df.head()

###Convert Totalcharges to float 
df['TotalCharges']=df['TotalCharges'].astype(str).astype(float)








def plot_corr(df):
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
sns.scatterplot(df.TotalCharges, df.tenure, df.nr_services)

df['MonthlyCharges']=df['MonthlyCharges'].astype(int)


plot_corr(df)

df = df.drop(["Churn"], 1)
df = df.drop(["customerID"], 1)

df1=df.dropna()
df1.isnull().sum()

kmeans = KMeans(n_clusters=4)
kmeans.fit(df1)

normalized_vectors = preprocessing.normalize(df1)
scores = [KMeans(n_clusters=i+2).fit(normalized_vectors).inertia_ for i in range(10)]
sns.lineplot(np.arange(2, 12), scores)
plt.xlabel('Number of clusters')
plt.ylabel("Inertia")
plt.title("Inertia of Cosine k-Means versus number of clusters")


normalized_kmeans = KMeans(n_clusters=4)
normalized_kmeans.fit(normalized_vectors)


#####DBSCAN###############


min_samples = df1.shape[1]+1 #  Rule of thumb; number of dimensions D in the data set, as minPts ≥ D + 1
dbscan = DBSCAN(eps=3.5, min_samples=min_samples).fit(df1)



#######PCA - Visualization#######################

from sklearn.decomposition import PCA

def prepare_pca(n_components, data, kmeans_labels):
    names = ['x', 'y', 'z']
    matrix = PCA(n_components=n_components).fit_transform(data)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.rename({i:names[i] for i in range(n_components)}, axis=1, inplace=True)
    df_matrix['labels'] = kmeans_labels
    
    return df_matrix

pca_df = prepare_pca(3, df1, normalized_kmeans.labels_)
sns.scatterplot(x=pca_df.x, y=pca_df.y, hue=pca_df.labels, palette="Set2")

def plot_3d(df, name='labels'):
    iris = px.data.iris()
    fig = px.scatter_3d(df, x='x', y='y', z='z',
                  color=name, opacity=0.5)
    

    fig.update_traces(marker=dict(size=3))
    fig.show()
pca_df = prepare_pca(3, df1, normalized_kmeans.labels_)
plot_3d(pca_df)




###########t-distributed Stochastic Neighbor Embedding (t-SNE) to visualize high dimensional data########
####This code doesn't work- can't see animated plots


def prepare_tsne(n_components, data, kmeans_labels):
    names = ['x', 'y', 'z']
    matrix = TSNE(n_components=n_components).fit_transform(data)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.rename({i:names[i] for i in range(n_components)}, axis=1, inplace=True)
    df_matrix['labels'] = kmeans_labels
    
    return df_matrix

tsne_3d_df = prepare_tsne(3, df1, kmeans.labels_)
plot_animation(tsne_3d_df, 'kmeans', 'kmeans')


tsne_3d_df1['normalized_kmeans'] = normalized_kmeans.labels_
plot_animation(tsne_3d_df1, 'normalized_kmeans', 'normalized_kmeans')


tsne_3d_df1['dbscan'] = dbscan.labels_
plot_animation(tsne_3d_df, 'dbscan', 'dbscan')


#####Evaluation##########################


kmeans = KMeans(n_clusters=4).fit(df1)

normalized_vectors = preprocessing.normalize(df1)
normalized_kmeans = KMeans(n_clusters=4).fit(normalized_vectors)

min_samples = df1.shape[1]+1 #  Rule of thumb; number of dimensions D in the data set, as minPts ≥ D + 1
dbscan = DBSCAN(eps=3.5, min_samples=min_samples).fit(df1)

print('kmeans: {}'.format(silhouette_score(df1, kmeans.labels_, metric='euclidean')))
print('Cosine kmeans: {}'.format(silhouette_score(normalized_vectors, normalized_kmeans.labels_, metric='cosine')))
print('DBSCAN: {}'.format(silhouette_score(df1, dbscan.labels_, metric='cosine')))

####Cosine based K-means outperfromed K-means.DBSCAN also performs well

######What makes Cluster Unique?###################
##One way to see the differences between clusters is to take the average value of each cluster and visualize it.

# Setting all variables between 0 and 1 in order to better visualize the results

scaler=MinMaxScaler()

df_scaled= pd.DataFrame(scaler.fit_transform(df1))
df_scaled.columns = df1.columns
df_scaled['dbscan'] = dbscan.labels_

# df = load_preprocess_data()
df1['dbscan'] = dbscan.labels_
tidy = df_scaled.melt(id_vars='dbscan')
fig, ax = plt.subplots(figsize=(15, 5))
sns.barplot(x='dbscan', y='value', hue='variable', data=tidy, palette='Set3')
plt.legend([''])
# plt.savefig("mess.jpg", dpi=300)
plt.savefig("dbscan_mess.jpg", dpi=300)

#####Go with the above approach if the number of variables are less than 10 as it would be difficult to visualize and interpret


#####Variance within variables and between clusters####

##What I essentially do is group datapoints by cluster and take the average. Then, I calculate the standard deviation between those values for each variable. Variables with a high standard deviation indicate that there are large differences between clusters and that the variable might be important.

df_mean = df_scaled.loc[df_scaled.dbscan!=-1,:].groupby('dbscan').mean().reset_index()

df_mean

# Setting all variables between 0 and 1 in order to better visualize the results
# df = load_preprocess_data().drop("Churn", 1)
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df1))
df_scaled.columns = df1.columns
df_scaled['dbscan'] = dbscan.labels_

# Calculate variables with largest differences (by standard deviation)
# The higher the standard deviation in a variable based on average values for each cluster
# The more likely that the variable is important when creating the cluster
df_mean = df_scaled.loc[df_scaled.dbscan!=-1, :].groupby('dbscan').mean().reset_index()

results = pd.DataFrame(columns=['Variable', 'Std'])
for column in df_mean.columns[1:]:
    results.loc[len(results), :] = [column, np.std(df_mean[column])]
selected_columns = list(results.sort_values('Std', ascending=False).head(7).Variable.values) + ['dbscan']

# Plot data
tidy = df_scaled[selected_columns].melt(id_vars='dbscan')
fig, ax = plt.subplots(figsize=(15, 5))
sns.barplot(x='dbscan', y='value', hue='variable', data=tidy, palette='Set3')
plt.legend(loc='upper right')
plt.savefig("dbscan_results.jpg", dpi=300)


#####Taken a reference from the following link

https://towardsdatascience.com/cluster-analysis-create-visualize-and-interpret-customer-segments-474e55d00ebb

