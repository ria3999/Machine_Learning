#%%
'''here all the modules and functions are being imported to the code'''
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings
import time
import sys
import os

#%%

'''depreciation is the discouragement of use of some terms,
feature,design or practice typically because it has been 
superseded or is no longer efficient or safe without removing 
or prohibiting its use.These practices may be removed from a 
release'''
DeprecationWarning('ignore')

#%%
'''ignoring all types of warnings and instead displays the given msg'''
warnings.filterwarnings('ignore',message="error warnings unwanted")

#%%
'''this function returns current working directory'''
s=os.getcwd()

#%%
'''to change directory'''
os.chdir('C:/machine learning')

#%%
'''to get the list of all files and directories in the specified
file'''
os.listdir()
#%%

'''to read .csv files .this function is in pandas'''
df=pd.read_csv('train.csv')
#%%

'''to return first 5 rows of the read file '''
df.head()

#%%
'''gives statistical info'''
df.describe()

#%%
'''sum of the number of values which are null in each attribute'''
df.isnull().sum
#%%
'''just a glimpse of table'''
df


#%%
'''last 5 rows'''
df.tail()


#%%
'''1 random row if not given any value as arguments'''
df.sample()

#%%
'''distplot is a way to have a quick look at univariate
 (having one variate or variable quantity)distribution'''
'''pandas dropna() method allows the user to analyze and drop
 rows/columns with null values'''
import seaborn as sns
sns.distplot(df.Fare.dropna())

#%%
'''splitting the data into train and test which is being pointed by df.
it will be trained on 80% data so the test_size is 0.2 and it will 
start splitting from 12th row starts'''
train,test=train_test_split(df,test_size=0.2,random_state=12)


#%%
# del df


#%%
'''return no. of rows and no of columns '''
train.shape

#%%
'''splitting data using position is possible by using iloc()'''
def train_test(df):
    length=len(df)
    train=df.iloc[:712,:]
    test=df.iloc[712:,]
    return train,test
    

#%%
train.head()



#%%
'''specifies ticket'''
df['Ticket']


#%%
'''loc is label-based,which means that you have to specify 
rows and columns based on their row and column labels. iloc 
is integer index based so you have to specify rows and 
columns by their integer index.'''
df.iloc[:,0:9]
df.loc[:,['Age','Parch']]

#%%
train.isnull().sum()

#%%
'''to calculate mean'''
mean_age=train.Age.mean()


#%%
'''to put mean age in empty or null areas'''
train['Age']=train['Age'].fillna(mean_age,inplace=True)
df=train.copy()

#%%
'''fill the null columns of age with mean'''
def fill_age(df):
    mean=29.67
    df['Age'].fillna(mean,inplace=True)
    return df

#%%
def fill_Embarked(df):
    df.Embarked.fillna('S',inplace=True)
    return(df)


#%%
def label_encoder(df):
    from sklearn.preprocessing import LabelEncoder
    label=LabelEncoder()
    df['Sex']=label.fit_transform(df['Sex'])
    df['Embarked']=label.fit_transform(df['Embarked'])
    return df


#%%
def encode_feature(df):
    df=fill_age(df)
    df=fill_Embarked(df)
    df=label_encoder(df)
    return df

#%%
train=encode_feature(train)
test=encode_feature(test)

#%%
def x_and_y (df):
    x=df.drop(['Survived','PassengerId','Cabin','Name','Ticket'],axis=1)
    y=df['Survived']
    return x,y


#%%
x_train,y_train=x_and_y(train)
x_test,y_test=x_and_y(test)

#%%
log_model= LogisticRegression()
log_model.fit(x_train,y_train)
prediction=log_model.predict(x_train)
score=accuracy_score(y_train,prediction)
print(score)
#%%
from sklearn.model_selection import cross_val_score
cv=cross_val_score(log_model,x_train,y_train,cv=5)
print(cv.mean())

#%%
from sklearn.metrics import precision_score,recall_score

#%%
from sklearn.metrics  import confusion_matrix
x_test_prediction=log_model.predict(x_train)
confusion_matrix(y_train,x_test_prediction)
from sklearn.metrics import precision_score,recall_score
recall=recall_score(y_train,x_test_prediction)
precision=precision_score(y_train,x_test_prediction)
print(f"recall is {recall}")
print(f"precision is{precision}")

#%%
from sklearn.metrics import roc_curve
prob=log_model.predict_proba(x_train)
fpr,tpr,thresholds=roc_curve(y_train,prob[:,1])


#%%
import matplotlib.pyplot as plt
plt.plot(fpr,tpr,linewidth=2)
plt.plot([0,1],[0,1])
plt.axis([0,1,0,1])


#%%
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
sns.set_style('whitegrid')
sns.distplot(tips.total_bill,hist=True,rug=True,bins=10)
sns.scatterplot(x='total_bill',y='tip',data=tips,hue='sex')
plt.show()

#%%
tips.describe()

#%%
sns.boxplot(x=tips.total_bill,orient='v')

#%%
