# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:53:42 2020

@author: Smita Gavandi
"""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt # data visualization
    from pandas import read_csv
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeRegressor

df=pd.read_csv("D:\\R Excel Sessions\\Assignments\\Decision Trees\\Fraud_check.csv")

df.dtypes
df.head
df.shape
df.info()
df.describe
df.columns

####Create a category treating those who have taxable_income <= 30000 as "Risky" and others are "Good"

#According to question i m adding a column 0r feature as fraud have class (risky, good) values based on Taxable.income
fraud = [] 
for value in df["Taxable.Income"]: 
    if value <= 30000: 
        fraud.append("risky") 
    else: 
        fraud.append("good") 
       
df["fraud"] = fraud    

df.head

from sklearn.model_selection import train_test_split

df1=df.drop(columns=['Taxable.Income'],axis=1)

df1.columns


###Frequncy counts of categorical variables
col_names=['Undergrad', 'Marital.Status','Urban', 'fraud']

for col in col_names:
    print(df1[col].value_counts())


####Check missing values
    
df1.isnull().sum()

# Analyse missing data
# Create table for missing data analysis
def draw_missing_data_table(df):
    total = df1.isnull().sum().sort_values(ascending=False)
    percent = (df1.isnull().sum()/df1.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

draw_missing_data_table(df1)

####Declare target variable and precitors

x=df1.drop(columns='fraud',axis=1)
y=df1['fraud']

x
y

type(x)
type(y)


# display categorical variables

#categorical = [col for col in x.columns if x[col].dtypes == 'O']

#categorical
#categorical.dtypes

#categorical=pd.DataFrame(columns=['Undergrad', 'Marital.Status', 'Urban'])

# display numerical variables

#numerical = [col for col in x.columns if x[col].dtypes != 'O']

#numerical

####Converting categorical columns to category



##Encode Categorical variables

from sklearn.preprocessing import LabelEncoder

# creating instance of labelencoder
labelencoder = LabelEncoder()

# Assigning numerical values and storing in another column
#categorical[['Undergrad', 'Marital.Status', 'Urban']] = labelencoder.fit_transform(categorical[['Undergrad', 'Marital.Status', 'Urban']])
#categorical

x['Undergrad1'] = labelencoder.fit_transform(x['Undergrad'])
x['Marital.Status1'] = labelencoder.fit_transform(x['Marital.Status'])
x['Urban1'] = labelencoder.fit_transform(x['Urban'])

x2=x.drop(columns=['Undergrad','Marital.Status','Urban'],axis=1)


###Split data into training and test dataset

from sklearn.model_selection import train_test_split

x2_train,x2_test,y_train,y_test = train_test_split(x2,y,test_size=0.30,random_state=42)

x2_train
x2_train.shape
x2_test
x2_test.shape
y_train
y_train.shape
y_test


#Decision Tree Classifier with criterion gini index

# import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier

# instantiate the DecisionTreeClassifier model with criterion gini index

clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)

# fit the model
clf_gini.fit(x2_train, y_train)

#Predict the Test set results with criterion gini index

y_pred_gini = clf_gini.predict(x2_test)


#Check accuracy score with criterion gini index

from sklearn.metrics import accuracy_score

print('Model accuracy score with gini index:{0:0.4f}'.format(accuracy_score(y_test,y_pred_gini)))

#Here, y_test are the true class labels and y_pred_gini are the predicted class labels in the test-set.

####Compare Train and Test Accuracy for overfitting

y_pred_train_gini=clf_gini.predict(x2_train)

print('Training set accuracy score :{0:0.4f}'.format(accuracy_score(y_train,y_pred_train_gini)))

###Check for overfitting and Underfitting

###Print the scores on Training and test dataset

print('Training set score :{0:0.4f}'.format(clf_gini.score(x2_train,y_train)))
print('Test set score :{0:0.4f}'.format(clf_gini.score(x2_test,y_test)))

#Here, the training-set accuracy score is 0.8048 while the test-set accuracy to be 0.7889. These two values are quite comparable. So, there is no sign of overfitting.


plt.figure(figsize=(12,8))
from sklearn import tree
tree.plot_tree(clf_gini.fit(x2_train,y_train))


 #Decision Tree Classifier with criterion entropy

#Ininitiate decision tree classifier model with criterian entrphy
 
clf_en=DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)

###fit the model

clf_en.fit(x2_train,y_train)

#Predict the Test set results with criterion entropy

y_pred_en = clf_en.predict(x2_test)

#Check accuracy score with criterion entropy

from sklearn.metrics import accuracy_score
print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test, y_pred_en)))

#Compare Train test accuracy

y_pred_train_en=clf_en.predict(x2_train)
y_pred_train_en

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_en)))

#Check for overfitting and underfitting

print('Training set score :{0:0.4f}'. format(clf_en.score(x2_train,y_train)))

print('Test set score :{0:0.4f}'. format(clf_en.score(x2_test,y_test)))

#We can see that the training-set score and test-set score is same as above. The training-set accuracy score is 0.8024 while the test-set accuracy to be 0.7889. These two values are quite comparable. So, there is no sign of overfitting.

#Visualize decision-trees
plt.figure(figsize=(12,8))
from sklearn import tree
tree.plot_tree(clf_en.fit(x2_train, y_train)) 


####Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_en)
print('Confusion matrix\n\n', cm)


####Classification report

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_en))


##############Check the model accuracy using XGBoost classifier#########################

#https://stackoverflow.com/questions/35139108/how-to-install-xgboost-in-anaconda-python-windows-platform
####To install xgboost run the following command in Anaconda command prompt

#conda install -c anaconda py-xgboost

import xgboost as xgb


####Use grid search to find number of trees(n_estimators)

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

model = XGBClassifier()
n_estimators = range(10, 100, 10)
param_grid = dict(n_estimators=n_estimators)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(x2, y)
print(grid.best_score_)
print(grid.best_params_)

####Best value for n_estimators is 20

#####Using grid search , we will find the best value for the depth of the tree

model = XGBClassifier()

max_depth=range(2, 12, 1)
param_grid = dict(max_depth=max_depth)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(x2,y)
print(grid.best_score_)
print(grid.best_params_)



###Build the model using n_estimators as 20 and max_depth as 2

xgb1 = XGBClassifier(objective ='reg:logistic', learning_rate = 0.1,
                max_depth = 2, n_estimators = 20)

xgb1.fit(x2_train,y_train)
train_pred =xgb1.predict(x2_train)

import numpy as np
train_acc = np.mean(train_pred==y_train) 
print(train_acc)


test_pred=xgb1.predict(x2_test)
test_acc=np.mean(test_pred==y_test)
print(test_acc)


#Variable importance plot

from xgboost import plot_importance
plot_importance(xgb1)


#####Using XGboost classifier , Training accuracy is 80% and Testing accuracy is also 80%

####################Adaboost Classifier############################

###Here 

#The most important parameters are base_estimator, n_estimators, and learning_rate.

#base_estimator is the learning algorithm to use to train the weak models. This will almost always not needed to be changed because by far the most common learner to use with AdaBoost is a decision tree – this parameter’s default argument.
#n_estimators is the number of models to iteratively train.
#learning_rate is the contribution of each model to the weights and defaults to 1. Reducing the learning rate will mean the weights will be increased or decreased to a small degree, forcing the model train slower (but sometimes resulting in better performance scores).
#loss is exclusive to AdaBoostRegressor and sets the loss function to use when updating weights. This defaults to a linear loss function however can be changed to square or exponential.



from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(n_estimators=50, 
                           learning_rate=1,
                           random_state=40)
model.fit(x2_train,y_train)
train_pred=model.predict(x2_train)

import numpy as np
train_acc = np.mean(train_pred==y_train) 
print(train_acc)

test_pred=model.predict(x2_test)
test_acc=np.mean(test_pred==y_test)
print(test_acc)


