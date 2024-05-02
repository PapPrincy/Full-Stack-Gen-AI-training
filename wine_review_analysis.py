#!/usr/bin/env python
# coding: utf-8

# # Importing libraries and dataset

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine #importing dataset from sklearn
wine_data = load_wine()
#loadung the dataset into dataframe usung pandas
df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
df['target']=wine_data.target
#displaying the first 5 rows
df.head()


# # Data Preprocessing with pandas

# In[2]:


df.info()


# In[3]:


df.shape


# In[4]:


df.isnull().sum()


# In[5]:


df.describe()


# In[6]:


df.corr()


# # Descriptive analysis with Numpy

# In[7]:


# Calculating mean, median, and standard deviation for each feature
mean = np.mean(wine_data.data, axis=0)
median = np.median(wine_data.data, axis=0)
std_dev = np.std(wine_data.data, axis=0)
min_value=np.min(wine_data.data,axis=0)
max_value=np.max(wine_data.data, axis=0)
percentile_25=np.percentile(wine_data.data,25,axis=0)
percentile_75=np.percentile(wine_data.data,75,axis=0)
'''Calculating mean,median and standard deviation for each feature'''

df1=pd.DataFrame({"Feature": wine_data.feature_names,
                  "Mean":mean,
                  "Median":median,
                   "std_dev":std_dev,
                "25th percentile":percentile_25,
                "75th percentile":percentile_75 })
print(df1)


# # Matrix calculation with Numpy
# 

# In[8]:


wine_data_matrix = wine_data.data
#Displaying the shape of the dataset without including the target column
print("Matrix representation of the wine dataset and its shape:\n")
print(wine_data_matrix)
print("\nShape of the matrix:", wine_data_matrix.shape)


# In[9]:


print("Calculating the Covariance...\n")
np.cov(wine_data_matrix)


# In[10]:


print("Calculating the correlation coefficient...")
np.corrcoef(wine_data_matrix)


# In[11]:


print("Calculating the transpose of the matrix....\n")
wine_data_matrix.T


# # Data Visualization with Matplotlib

# In[12]:


import matplotlib.pyplot as plt
print("\n**********Histogram for distribution of alcohol content*********\n")
plt.hist(df['alcohol'], bins=10, color='blue', edgecolor='red')
plt.title('Distribution of Alcohol Content in Wine')
plt.xlabel('alcohol Content')
plt.ylabel('Frequency')
plt.show()


# In[13]:


import matplotlib.pyplot as plt

# Extracting attributes for the scatter plot
alcohol = df['alcohol']
flavanoids = df['flavanoids']
wine_class = df['target']  # For coloring points based on wine class

# Create a scatter plot
plt.figure(figsize=(10,6))
plt.scatter(alcohol, flavanoids, c=wine_class, cmap='cividis', alpha=0.7)
plt.xlabel('Alcohol Content')
plt.ylabel('Flavanoids')
plt.title('Scatter Plot of Alcohol Content vs Flavanoids')

# Add a colorbar to show wine class
cbar = plt.colorbar()
cbar.set_label('Wine Class')

# Show the plot
plt.grid(True)
plt.show()


# In[ ]:




