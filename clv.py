
# coding: utf-8

#  # Prediction of Customer Lifetime Value (CLV) 
#    # Bibobra Alabrah

# BUSINESS PROBLEM
# 
# A company wants to know the lifetime value of customers in terms of how much money they will likely bring to the company based on their first few purchase history.
# 
# 
# GOAL
# 
# The goal of this project is to build a predictive model that estimates the customer lifetime value (CLV) for new customers using past purchase history of existing customers.

# In[1]:


# Import dependences

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics


# In[2]:


# Load the data set and view the summary statistics

purchase_history = pd.read_csv("history.csv")


# ## Exploratory Data Analysis

# In[3]:


# View the data types
purchase_history.dtypes


# The dataset consists of the customer ID, the amount the customer spent on your website for the first months of his relationship with your business and his ultimate life time value ( say 3 years worth)

# In[4]:


# View the dimension of the data set
purchase_history.shape


# There are 100 customers for this dataset

# In[5]:


# View the first few records of the data
purchase_history.head()


# In[6]:


# View the last few records
purchase_history.tail()


# ## Select the best features using Correlation Analysis

# In[7]:


purchase_history.corr()['CLV']


# We can see that the months do show strong correlation to the target variable (CLV). That should give us confidence that we can build a strong model to predict the CLV, but the customer ID has no correlation or whatsoever, so we remove it.

# ## Check for missing values

# In[8]:


purchase_history.isnull().sum()


# There are no missing values.

# # Data Cleaning

# In[9]:


# We now remove the customer id feature
clean = purchase_history.drop("CUST_ID",axis=1)


# In[10]:


# Let us confirm that the data looks exactly as desired
clean.head()


# ## Split the data into a train and validation set
# 
# Let us split the data into training and testing in the ratio of 80:20
# 
# But first, we have to drop the target variable(CLV) to form the predictors

# In[11]:


predictors = clean.drop("CLV",axis=1)
target = clean.CLV

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, target, test_size=.2)
print( "Predictor - Training : ", pred_train.shape, "Predictor - Testing : ", pred_test.shape )


# ## Build and Test Model
# We build a Linear Regression equation for predicting CLV and then check its accuracy by predicting against the test dataset

# In[12]:


# Build model on training data

# instantiate the model
LR_model = LinearRegression()

# Fit the model
LR_model.fit(pred_train,tar_train)

print("Coefficients: \n", LR_model.coef_)
print("Intercept:", LR_model.intercept_)


# In[13]:


# Let us test this model on the test data set

predictions = LR_model.predict(pred_test)
predictions

# Check the accuracy of the predictions
sklearn.metrics.r2_score(tar_test, predictions)


# It shows a 88% accuracy. This is a good model for predicting CLV for new customers

# ## Predict for a new Customer
# Let us say we have a new customer who in his first 3 months have spend 300,100,250 on purchases. Let us use the model to predict his CLV.

# In[14]:


new_data = np.array([300,100,250,0,0,0]).reshape(1, -1)
new_data


# In[15]:


new_pred = LR_model.predict(new_data) 
print("The CLV for the new customer is : $",new_pred[0])

