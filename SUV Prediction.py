#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset=pd.read_csv("C:/Users/Lenovo/suv_data.csv")


# In[3]:


dataset.head()


# In[5]:


x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values


# In[6]:


y


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


# In[11]:


from sklearn.preprocessing import StandardScaler


# In[12]:


sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[13]:


from sklearn.linear_model import LogisticRegression


# In[15]:


classifier=LogisticRegression (random_state=0)
classifier.fit(x_train,y_train)


# In[16]:


y_pred=classifier.predict(x_test)


# In[18]:


from sklearn.metrics import accuracy_score


# In[19]:


accuracy_score(y_test,y_pred)*100


# In[ ]:




