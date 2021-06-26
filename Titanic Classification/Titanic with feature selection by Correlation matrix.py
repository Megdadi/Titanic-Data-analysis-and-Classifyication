import pandas as pd
import numpy as np
import random as rnd
import os

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

path= "D:/Titanic Classification/"
os.listdir(path)
df_train= pd.read_csv(path+ 'train.csv')
df_test= pd.read_csv(path+ 'test.csv')
combine = [df_train, df_test]
df_train.columns
############################################
df_train.info()
'''
We observe the name does not affect on target
We observe the Ticket does not affect on target

so we will drop this features
'''
df_train=df_train.drop(['Name','Ticket'],axis=1)
df_test=df_test.drop(['Name','Ticket'],axis=1)
df_test.shape
df_train.shape
###########################################
# Which features contain blank, null or empty values?
df_train.info()
'''
We observe the Age has missing values(891 - 714 = 177)
so we will Impute mean value
We observe the Cabin contain (891 - 204 = 687 missing values).
'''
df_train=df_train.drop(['Cabin'],axis=1)
df_test=df_test.drop(['Cabin'],axis=1) 

from numpy import nan
mean_Age=df_train['Age'].mean()

values = {'Age':mean_Age}
df_train=df_train.fillna(value=values)
values = {'Embarked':'C'}
df_train=df_train.fillna(value=values)


mean_Age=df_test['Age'].mean()
values = {'Age':mean_Age}
df_test=df_test.fillna(value=values)

mean_Fare=df_test['Fare'].mean()
values = {'Fare':mean_Age}
df_test=df_test.fillna(value=values)

df_test.info()
df_train.info()
# there is no missing values
##############################################
# Which features are categorical?
# Which features are numerical?
df_train.describe()# only numerical features 
df_train.describe(include=['O'])# categiorical features types
'''
sex (Nominal Data) has two categorical that has (male, female) and does not
have rank (Nominal Data)  For example, Gender (Male/Female/Other),
 Age Groups (Young/Adult/Old), etc
### One-Hot encoding ######
'''
DataDummies = pd.get_dummies(df_train['Sex'])
df_train=df_train.drop(['Sex'],axis=1)

df_train=pd.concat([df_train, DataDummies],axis=1)
############
DataDummies = pd.get_dummies(df_test['Sex'])
df_test=df_test.drop(['Sex'],axis=1)

df_test=pd.concat([df_test, DataDummies],axis=1)
#################################################################
'''
# Age is ordinal data that a kind of categorical data with a set order
# '''
### train
# Let us create Age bands and determine correlations with Survived
df_train['AgeBand'] = pd.cut(df_train['Age'], 4)
df_train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

# Let us replace Age with ordinals based on these bands.
    
df_train.loc[ df_train['Age'] <= 20, 'Age'] = 0
df_train.loc[(df_train['Age'] > 20) & (df_train['Age'] <= 40), 'Age'] = 1
df_train.loc[(df_train['Age'] > 40) & (df_train['Age'] <= 60), 'Age'] = 2
df_train.loc[ df_train['Age'] > 60, 'Age'] = 3
compare= pd.concat([df_train['Age'], df_train['AgeBand']],axis=1)
df_train= df_train.drop(['AgeBand'],axis=1)
df_train.head(10)

#####################################

### test
#Let us create Age bands and determine correlations with Survived
df_test['AgeBand'] = pd.cut(df_test['Age'], 4)

# Let us replace Age with ordinals based on these bands.
    
df_test.loc[ df_test['Age'] <= 20, 'Age'] = 0
df_test.loc[(df_test['Age'] > 20) & (df_test['Age'] <= 40), 'Age'] = 1
df_test.loc[(df_test['Age'] > 40) & (df_test['Age'] <= 60), 'Age'] = 2
df_test.loc[ df_test['Age'] > 60, 'Age'] = 3
compare= pd.concat([df_test['Age'], df_test['AgeBand']],axis=1)
df_test= df_test.drop(['AgeBand'],axis=1)
df_test.head(10)
#################################################################
# replase values by dictionary

# df_train.describe(include=['O'])# categiorical features types
# df_train= df_train.drop([61])# Embarked in 61 index= 29.6991
# df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()


# Embarked_Dict =   { 'S':0, 'Q':1, 'C':2 }                       
# df_train['Embarked'] = df_train.Embarked.map(Embarked_Dict)
# df_test['Embarked'] = df_test.Embarked.map(Embarked_Dict)
# df_train.dropna(inplace=True)

# df_test.info()
# df_train.info()
############################ handle Embarked by Target Encoding ################################
encodings = df_train.groupby('Embarked')['Survived'].mean().reset_index()
df_train = df_train.merge(encodings, how='left', on='Embarked')
df_train.drop('Embarked', axis=1, inplace=True)
df_train.columns
###### Relationship with numerical variables and handle anomaly data #########

import matplotlib.pyplot as plt # Matlab-style plotting
#scatter plot grlivarea/saleprice
df_train.columns
var = 'Fare'# متغير المساحة
data = pd.concat([df_train['Survived_x'],df_train[var]], axis=1)
# concat : وضع العمودين مع بعض
print(data[:5])

data.plot.scatter(x=var, y='Survived_x',  ylim=data)
a=df_train[df_train['Fare'] > 300].index
points=a
print(points)
df_train = df_train.drop(points)

#################################### splite the data and scaling
X = df_train.drop('Survived_x',axis=1)
y = df_train['Survived_x']
X.shape
y.shape

# ###############  Correlation matrix (heatmap style) ###########
numberofcolumns= len(X.columns) #  number of columns
from scipy.stats import pearsonr
correlation=[]
aaa= X.columns
aaa=pd.Series(aaa)
# corr, _ = pearsonr(X['PassengerId'], y)
i=0
for i in range(numberofcolumns):
    corr, _ = pearsonr(X.iloc[:,i], y)
    correlation.append(corr)
    
correlation=pd.Series(correlation)
correlation=pd.concat([correlation,aaa],axis=1)
correlation.sort_values(0,ascending=True)

X=X.drop([ 'PassengerId','Parch'],axis=1)




##### scaling data
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#ss = StandardScaler()
#X = ss.fit_transform(X)
minmax = MinMaxScaler()
X = minmax.fit_transform(X)
##############
### Perform train and test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=128)
X_train.shape
y_train.shape
################################# Logistic Regression Classifier ##################
from sklearn.linear_model import LogisticRegression


lrClassifier = LogisticRegression()
lrClassifier.fit(X_train,y_train)
### Prediction on test data
prediction = lrClassifier.predict(X_test)
prediction[:10] # predict for the first ten rows
print(y_test[:10])
## Measure accuracy of the classifier
from sklearn.metrics import accuracy_score

acc_log= accuracy_score(y_true=y_test, y_pred=prediction)

########################### Support Vector Machines ##########################
from sklearn.svm import SVC, LinearSVC

svc = SVC()
svc.fit(X_train,y_train)
prediction = svc.predict(X_test)
prediction[:10] # predict for the first ten rows
print(y_test[:10])
## Measure accuracy of the classifier

acc_svc= accuracy_score(y_true=y_test, y_pred=prediction)

############################ Random Forest #################################
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
prediction = random_forest.predict(X_test)
prediction[:10] # predict for the first ten rows
print(y_test[:10])
## Measure accuracy of the classifier

acc_random_forest= accuracy_score(y_true=y_test, y_pred=prediction)
######################### Decision Tree #####################################
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
prediction = decision_tree.predict(X_test)
prediction[:10] # predict for the first ten rows
print(y_test[:10])
## Measure accuracy of the classifier

acc_decision_tree= accuracy_score(y_true=y_test, y_pred=prediction)
########################## KNN ##########################################
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)
prediction[:10] # predict for the first ten rows
print(y_test[:10])
## Measure accuracy of the classifier

acc_knn= accuracy_score(y_true=y_test, y_pred=prediction)
############################ Gaussian Naive Bayes ########################
from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()
NB.fit(X_train, y_train)
prediction = NB.predict(X_test)
prediction[:10] # predict for the first ten rows
print(y_test[:10])
## Measure accuracy of the classifier

acc_NB=accuracy_score(y_true=y_test, y_pred=prediction)
###################### Perceptron #######################################
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, y_train)
prediction = perceptron.predict(X_test)
## Measure accuracy of the classifier

acc_perceptron= accuracy_score(y_true=y_test, y_pred=prediction)
######################## SGD ##########################################
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
perceptron = sgd.predict(X_test)
## Measure accuracy of the classifier

acc_sgd=accuracy_score(y_true=y_test, y_pred=prediction)
############### Model evaluation ######################################
############### Model evaluation ######################################
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_NB, acc_perceptron, 
              acc_sgd,  acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
