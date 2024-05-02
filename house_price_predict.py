#!/usr/bin/env python
# coding: utf-8

# # Importing libraries and dataset

# In[1]:


import numpy as np
import pandas as pd
df=pd.read_csv("C:\\Users\\epvin\\OneDrive\\Desktop\\Gen AI-Training\\housing_price_dataset.csv")
df['target']= df['Price']
df.drop(['Price'],axis=1,inplace=True)
df.head()


# # Data Preprocessing with pandas

# In[2]:


df.info()


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.dtypes


# In[6]:


df['Neighborhood'].unique() #checking for values in the specific column


# In[7]:


d={'Rural': 0, 'Suburb': 1, 'Urban': 2}
df['Neighborhood'] = df['Neighborhood'].replace(d)
df.head()


# In[8]:


df.isnull().sum()


# In[9]:


df.shape


# In[10]:


df['target'].describe()


# # Descriptive analysis with NumPy
# 

# In[11]:


df.drop(['Neighborhood'],axis=1,inplace=True)


# In[12]:


# Calculating mean, median, and standard deviation for each feature
mean = np.mean(df, axis=0)
median = np.median(df, axis=0)
std_dev = np.std(df, axis=0)
min_value = np.min(df, axis=0)
max_value = np.max(df, axis=0)
percentiles = df.describe(percentiles=[0.25, 0.75]).loc[['25%', '75%']]

df1 = pd.DataFrame({
    "Feature": df.columns,
    "Mean": mean,
    "Median": median,
    "Std Dev": std_dev,
    "Min": min_value,
    "Max": max_value,
    "25th Percentile": percentiles.loc['25%'],
    "75th Percentile": percentiles.loc['75%']
})
df1


# # Matrix Calculation with Numpy

# In[13]:


#Displaying the shape of the dataset without including the target column
matrix_data=df.values
print("Matrix representation of the wine dataset and its shape:\n")
print(matrix_data)
print("\nShape of the matrix:", matrix_data.shape)


# In[14]:


print("Calculating the Covariance...\n")
first_10_rows=df.head(10)
np.cov(first_10_rows,rowvar=False)


# In[15]:


print("Calculating the correlation coefficient...")
first_10_rows=df.head(10)
np.corrcoef(first_10_rows,rowvar=False)


# In[16]:


print("Calculating the transpose of the matrix....\n")
matrix_data.T


# # Data Visualization with matplotlib and seaborn
# 

# In[17]:


import matplotlib.pyplot as plt

# Extracting attributes for the scatter plot
YearBuilt = df['YearBuilt']
price = df['target']
price_class = df['target']  # For coloring points 

# Create a scatter plot
plt.figure(figsize=(10,6))
plt.scatter(YearBuilt,price, c=price_class, cmap='cividis', alpha=0.7)
plt.xlabel('year build')
plt.ylabel('price')
plt.title('Scatter Plot of price vs year built')

# Add a colorbar to show wine class
cbar = plt.colorbar()
cbar.set_label('price Class')

# Show the plot
plt.grid(True)
plt.show()


# In[18]:


import seaborn as sns

# Set the style
sns.set(style="darkgrid")
plt.figure(figsize=(10, 6))
sns.histplot(df['target'], bins=30, kde=True,color='red')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Prices')
plt.show()


# # feature engineering and model training with scikit learn

# In[22]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Splitting the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#identifying numerical columns
num_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
scaler=StandardScaler()

# Fit and transform the numerical columns using the scaler
X_train_scaled = X_train.copy()
X_train_scaled[num_columns] = scaler.fit_transform(X_train[num_columns])

# Transform the numerical columns of test data using the scaler fitted on training data
X_test_scaled = X_test.copy()
X_test_scaled[num_columns] = scaler.transform(X_test[num_columns])

# Printing the first few rows of the scaled training data
print("Scaled Training Data:")
print(X_train_scaled.head())

# Printing the first few rows of the scaled test data
print("\nScaled Test Data:")
print(X_test_scaled.head())

# Fit and evaluate the model
model=LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("\n\nMean Squared Error:",mse)
rsquare=r2_score(y_test,y_pred)
print("\nR-squared score:",rsquare)


# # writing and loading the model in pickle

# In[20]:


import pickle
with open("c:\\Users\\epvin\\Regression_model.pickle", 'wb') as file:
                 pickle.dump(model, file)
print("Data saved successfully.")


# In[21]:


with open("c:\\Users\\epvin\\problem1_data_file.pickle", 'rb') as file:
     model = pickle.load(file)
print("Data loaded successfully.")


# In[ ]:




