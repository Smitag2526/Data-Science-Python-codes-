# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 13:58:36 2020

@author: Smita Gavandi
"""

#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns#Understanding my variables

df=pd.read_csv(r"D:\R Excel Sessions\Practice datasets\Usedcars_EDA\vehicles.csv")
pd.set_option('display.max_columns', None)
df.head()
df.shape
df.columns

df.nunique(axis=0)
df.info()


df.describe().apply(lambda s:s.apply(lambda x :format(x,'f')))

####To see unique values in the discrete variable
df.condition.unique()

####To reclassify the condition values

def clean_condition(row):
    
    good = ['good','fair']
    excellent = ['excellent','like new']       
    
    if row.condition in good:
        return 'good'   
    if row.condition in excellent:
        return 'excellent'    
    return row.condition# Clean dataframe
def clean_df(playlist):
    df_cleaned1 = df.copy()
    df_cleaned1['condition'] = df_cleaned1.apply(lambda row: clean_condition(row), axis=1)
    return df_cleaned1# Get df with reclassfied 'condition' column
df_cleaned1 = clean_df(df)
print(df_cleaned1.condition.unique())

#####Removing /reduntant valriables

df.columns

df_cleaned=df_cleaned.copy().drop(['url','image_url','region_url'],axis=1)

###Variable Selection(Remove columns that has 40% or more as Null)

NA_val = df_cleaned.isna().sum()
NA_val

def na_filter(na, threshold=.4):
    
    #only select variables that passees the threshold
    col_pass=[]
    for i in na.keys(): 
        if na[i]/df_cleaned.shape[0]<threshold:
            col_pass.append(i)
    return col_pass
df_cleaned = df_cleaned[na_filter(NA_val)]
df_cleaned.columns


###Removing Outliers

df_cleaned=df_cleaned[df_cleaned['price'].between(999.99,99999.00)]

df_cleaned=df_cleaned[df_cleaned['year']>1990]

df_cleaned=df_cleaned[df_cleaned['odometer']<899999.00]

###Calculate Correlation Matrix

corr= df_cleaned.corr()

sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,annot=True,cmap=sns.diverging_palette(220, 20, as_cmap=True))


###Scatterplot

df_cleaned.plot(kind='scatter',x='odometer',y='price')

df_cleaned.plot(kind='scatter', x='year',y='price')

sns.pairplot(df_cleaned)
    
df_num = df_cleaned[['price','year','odometer','lat','long']]
df_num1=df_num[:10000]


sns.pairplot(df_num1)

df_num.isna().sum()

#####Since variables 'lat' and 'long' have NA , we will replace it with 0
df_num['lat']=df_num['lat'].replace(np.nan,0)
df_num['long']=df_num['long'].replace(np.nan,0)

df_num.isna().sum()

sns.pairplot(df_num)


#####Histogram

df_cleaned['odometer'].plot(kind='hist',bins=50,figsize=(12,6),facecolor='grey',edgecolor='black')


df_cleaned['year'].plot(kind='hist',bins=10,figsize=(12,6),facecolor='grey',edgecolor='black')


#####Boxplot

df_cleaned.boxplot('price')

