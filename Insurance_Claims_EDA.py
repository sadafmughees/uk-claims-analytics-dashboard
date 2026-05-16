#!/usr/bin/env python
# coding: utf-8

# # Data Quality & Exploratory Data Analysis Report
# 
# ## Project:
# UK Insurance Claims Analytics System
# 
# ## Objective
# This report presents the exploratory data analysis (EDA) and data quality assessment performed on the synthetic UK claims dataset. The purpose of this analysis is to identify data patterns, validate data quality, and determine suitable features for machine learning model development.
# 

# In[ ]:


import pandas as pd


# In[12]:


df = pd.read_csv("claims_demo.csv")

df.head()


# # Dataset Overview
# 
# The dataset contains 500 synthetic insurance claim records generated using Python. The dataset includes claim details, claim amount, processing time, and approval status.

# In[13]:


#Dataset info
df.info()


# # Data Quality Assessment
# 
# The dataset structure was analyzed to identify missing values, incorrect data types, and inconsistencies.
# 
# ## Findings
# - No major missing values identified
# - Numerical and categorical columns successfully detected
# - Data formatting was consistent
# - Dataset suitable for analysis and machine learning workflows

# In[14]:


#Missing values
df.isnull().sum()


# In[15]:


#Statistics
df.describe()


# # Claim Status Analysis
# 
# The distribution of claim approval statuses was analyzed to understand approval and rejection trends.

# In[16]:


#Status analysis
df['Status'].value_counts()


# In[17]:


#Graph (important for report)
import matplotlib.pyplot as plt

df['Status'].value_counts().plot(kind='bar')

plt.show()


# In[19]:


df.columns


# In[22]:


#Claim Type analysis
df['Claim_Type'].value_counts()


# In[23]:


#Average claim amount
df['Claim_Amount_GBP'].mean()


# In[24]:


#Maximum claim amount
df['Claim_Amount_GBP'].max()


# In[25]:


#Average processing time
df['Processing_Time_Days'].mean()


# In[26]:


#Status Bar Chart
import matplotlib.pyplot as plt

df['Status'].value_counts().plot(kind='bar')

plt.title("Claim Status Distribution")

plt.xlabel("Status")

plt.ylabel("Count")

plt.show()


# ## Observation
# 
# Most claims were successfully approved, while rejected and pending claims represented a smaller proportion of the dataset.

# In[27]:


#Claim Type Chart
df['Claim_Type'].value_counts().plot(kind='bar')

plt.title("Claim Type Distribution")

plt.show()


# ## Observation
# 
# Different claim categories showed varying frequencies, indicating diverse insurance claim patterns across the dataset.

# In[28]:


#Claim Amount Histogram
df['Claim_Amount_GBP'].plot(kind='hist', bins=20)

plt.title("Claim Amount Distribution")

plt.show()


# ## Observation
# 
# Claim amounts showed moderate variation with a few high-value claims identified in the dataset.

# In[29]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


## Processing Time Analysis
plt.figure(figsize=(8,5))

df['Processing_Time_Days'].plot(
    kind='hist',
    bins=15
)

plt.title("Processing Time Distribution")

plt.xlabel("Days")

plt.ylabel("Frequency")

plt.show()


# ## Observation
# 
# Processing times varied across claims, suggesting operational differences in claim handling procedures.

# # Feature Selection for Machine Learning
# 
# The following features were identified as suitable for machine learning model development.
# 
# ## Selected Features
# - Claim_Type
# - Claim_Amount_GBP
# - Processing_Time_Days
# - Policy_Holder_City
# 
# ## Target Variable
# - Status

# # Conclusion
# 
# The dataset was successfully analyzed and validated for future dashboard development and machine learning workflows. The data quality was found to be satisfactory, and the selected features are suitable for predictive analytics applications.

# In[ ]:




