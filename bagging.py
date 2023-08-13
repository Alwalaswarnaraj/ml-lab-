#!/usr/bin/env python
# coding: utf-8

# In[85]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import  BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# In[86]:


import warnings
warnings.filterwarnings('ignore')


# In[87]:


da=pd.read_csv('D:/python/IRIS.csv')


# In[88]:


cols=['0.222222222','0.625','0.06779661','0.041666667']


# In[89]:


da.drop(cols, axis=1, inplace=True)


# In[90]:


headerlist=['Sepal Length','Sepal Width','Petal Length','Peatal Width','Species']

da.to_csv("IRIS2.csv", header=headerlist, index=False)


# In[91]:


da=pd.read_csv('IRIS2.csv')


# In[92]:


da


# In[93]:


da.head(2)


# In[94]:


x=da.iloc[:,[0,1,2,3]].values


# In[95]:


y=da.iloc[:,4].values


# In[96]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)


# In[97]:


single=GaussianNB()
single.fit(xtrain,ytrain)


# In[99]:


ypred=single.predict(xtest)
ypred


# In[101]:


acc=accuracy_score(ytest,ypred)*100
acc


# In[102]:


cm=confusion_matrix(ytest,ypred)
cm


# In[103]:


base_calss=GaussianNB()
num_class=50


# In[111]:


bag=BaggingClassifier(base_estimator=GaussianNB(),n_estimators=num_class,random_state=0)
bag.fit(xtrain,ytrain)


# In[112]:


res=model_selection.cross_val_score(bag,xtest,ytest,cv=10)
print(res.mean()*100)


# In[ ]:





# In[ ]:




