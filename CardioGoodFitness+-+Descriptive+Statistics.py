#!/usr/bin/env python
# coding: utf-8

# # Cardio Good Fitness Case Study - Descriptive Statistics
# The market research team at AdRight is assigned the task to identify the profile of the typical customer for each treadmill product offered by CardioGood Fitness. The market research team decides to investigate whether there are differences across the product lines with respect to customer characteristics. The team decides to collect data on individuals who purchased a treadmill at a CardioGoodFitness retail store during the prior three months. The data are stored in the CardioGoodFitness.csv file.
# 
# ### The team identifies the following customer variables to study: 
#   - product purchased, TM195, TM498, or TM798; 
#   - gender; 
#   - age, in years; 
#   - education, in years; 
#   - relationship status, single or partnered; 
#   - annual household income ; 
#   - average number of times the customer plans to use the treadmill each week; 
#   - average number of miles the customer expects to walk/run each week; 
#   - and self-rated fitness on an 1-to-5 scale, where 1 is poor shape and 5 is excellent shape.
# 
# ### Perform descriptive analytics to create a customer profile for each CardioGood Fitness treadmill product line.

# In[1]:


# Load the necessary packages

import numpy as np
import pandas as pd


# In[9]:


# Load the Cardio Dataset

mydata = pd.read_csv('C:\\Users\\hp\\Downloads\\CardioGoodFitness.csv')


# In[7]:


mydata.head()


# In[8]:


mydata.describe(include="all")


# In[7]:


mydata.info()


# In[14]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

mydata.hist(figsize=(20,30))


# In[20]:


import seaborn as sns

sns.boxplot(x="Gender", y="Age", data=mydata)


# In[21]:


pd.crosstab(mydata['Product'],mydata['Gender'] )


# In[22]:


pd.crosstab(mydata['Product'],mydata['MaritalStatus'] )


# In[24]:


sns.countplot(x="Product", hue="Gender", data=mydata)


# In[41]:


pd.pivot_table(mydata, index=['Product', 'Gender'],
                     columns=[ 'MaritalStatus'], aggfunc=len)


# In[42]:


pd.pivot_table(mydata,'Income', index=['Product', 'Gender'],
                     columns=[ 'MaritalStatus'])


# In[43]:


pd.pivot_table(mydata,'Miles', index=['Product', 'Gender'],
                     columns=[ 'MaritalStatus'])


# In[44]:


sns.pairplot(mydata)


# In[45]:


mydata['Age'].std()


# In[46]:


mydata['Age'].mean()


# In[50]:


sns.distplot(mydata['Age'])


# In[58]:


mydata.hist(by='Gender',column = 'Age')


# In[59]:


mydata.hist(by='Gender',column = 'Income')


# In[60]:


mydata.hist(by='Gender',column = 'Miles')


# In[62]:


mydata.hist(by='Product',column = 'Miles', figsize=(20,30))


# In[67]:


corr = mydata.corr()
corr


# In[66]:


sns.heatmap(corr, annot=True)


# In[96]:


# Simple Linear Regression


#Load function from sklearn
from sklearn import linear_model

# Create linear regression object
regr = linear_model.LinearRegression()

y = mydata['Miles']
x = mydata[['Usage','Fitness']]

# Train the model using the training sets
regr.fit(x,y)


# In[97]:


regr.coef_


# In[98]:


regr.intercept_


# In[ ]:


# MilesPredicted = -56.74 + 20.21*Usage + 27.20*Fitness

