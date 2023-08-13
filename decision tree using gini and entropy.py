#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import seaborn as sns


# In[7]:


da=pd.read_csv("D:/python/IRIS.csv")


# In[8]:


col_drop=['0.222222222','0.625','0.06779661','0.041666667']
da.drop(col_drop,axis=1,inplace=True)


# In[9]:


headerlist=['Sepal Length','Sepal Width','Petal Length','Peatal Width','Species']

da.to_csv("IRIS2.csv", header=headerlist, index=False)


# In[10]:


da.head(2)


# In[11]:


da=pd.read_csv("IRIS2.csv")


# In[12]:


da.head(2)


# In[13]:


sns.pairplot(da)


# In[15]:


x=da.iloc[:,[0,1,2,3]].values
print(x)


# In[16]:


y=da.iloc[:,4].values


# In[17]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)


# In[18]:


sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)


# In[19]:


dtree_gini=DecisionTreeClassifier(criterion='gini',random_state=100,max_depth=3,min_samples_leaf=5)


# In[20]:


dtree_gini.fit(xtrain,ytrain)


# In[22]:


ypred=dtree_gini.predict(xtest)
ypred


# In[23]:


acc=accuracy_score(ytest,ypred)*100
acc


# In[24]:


cm=confusion_matrix(ytest,ypred)


# In[25]:


cm


# In[33]:


fig,ax=plt.subplots(figsize=(6,6))
ax.imshow(cm)
ax.grid=False
ax.xaxis.set(ticks=(0,1,2),ticklabels=('predicted setosa','predicted versicolor','predicted virginica'))
ax.yaxis.set(ticks=(0,1,2),ticklabels=('actual setosa','actual versicolor','actualverginica'))
ax.set_ylim(2.5,-0.5)
for i in range(3):
    for j in range(3):
        ax.text(j,i,cm[i,j],ha='center',va='center',c='white')
        


# In[34]:


cr=classification_report(ytest,ypred)


# In[36]:


print(cr)


# In[38]:


tree.plot_tree(dtree_gini)


# In[39]:


dtree_en=DecisionTreeClassifier(criterion='entropy',random_state=100,max_depth=3,min_samples_leaf=5)


# In[40]:


dtree_en.fit(xtrain,ytrain)


# In[41]:


ypred1=dtree_en.predict(xtest)


# In[42]:


print(ypred1)


# In[43]:


acc1=accuracy_score(ytest,ypred1)*100
print(acc1)


# In[44]:


cm1=confusion_matrix(ytest,ypred1)
cm1


# In[45]:


tree.plot_tree(dtree_en)


# In[51]:


fig,ax=plt.subplots(figsize=(6,6))
ax.imshow(cm1)
ax.grid=False
ax.xaxis.set(ticks=(0,1,2),ticklabels=('predicted setosa','predicted versicolor','predicted virginica'))
ax.yaxis.set(ticks=(0,1,2),ticklabels=('actual setosa','ac versicolor','ac verginica'))
ax.set_ylim(2.5,-0.5)
for i in range(3):
    for j in range(3):
        ax.text(j,i,cm1[i,j],ha='center',va='center',c='black')


# In[ ]:




