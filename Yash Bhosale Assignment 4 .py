#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # Assignment - 04 (Simple Linear Regression)

# # Q1) Predict delivery time using sorting time
# ### Build a simple linear regression model by performing EDA and do necessary transformations and select the best model using R or Python.

# In[2]:


# import libraries
import warnings
warnings.filterwarnings('ignore')
import pandas as pd # used for data analysis and data maipulation  
import numpy as np # used for working with arrays
import seaborn as sns # used for visualization
from scipy.stats import norm # used statistics for normal distribution
from scipy.stats import skew # to handle skewness
import statsmodels.formula.api as smf # to Create a Model from a formula and dataframe
import matplotlib.pyplot as plt # used for visualization


# In[3]:


dataset=pd.read_csv(r"C:\Users\Yash Bhosale\Downloads\delivery_time.csv")
dataset


# #### EDA and Data Visualization

# In[4]:


dataset.info()


# In[5]:


dataset.describe()


# #### Feature Engineering

# In[6]:


# Renaming Columns
dataset=dataset.rename({'Delivery Time':'delivery_time', 'Sorting Time':'sorting_time'},axis=1)
dataset


# In[7]:


print(skew(dataset.delivery_time))


# In[8]:


sns.boxplot(dataset['delivery_time'], orient = 'h')


# In[9]:


sns.distplot(dataset['delivery_time'])


# In[10]:


print(skew(dataset.sorting_time))


# In[11]:


sns.boxplot(dataset['sorting_time'], orient='h')


# In[12]:


# The observations for Sorting Time lies nearly between 4 to 8.
# It means the Sorting Time data is symmetric and skewed.
# The median sorting time is approximately around 6.


# In[13]:


sns.distplot(dataset['sorting_time'])


# In[14]:


sns.pairplot(dataset)


# #### Transformation for Continuous Variable : 

# In[15]:


# log trasformation :
dataset['log_delivery_time']= np.log(dataset["delivery_time"])

fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax[0].hist(dataset['delivery_time']);ax[0].set_title("delivery_time")
ax[1].hist(dataset['log_delivery_time']);ax[1].set_title("Log_delivery_time")
plt.show()


# In[16]:


dataset['log_sorting_time']= np.log(dataset["sorting_time"])

fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax[0].hist(dataset['sorting_time']);ax[0].set_title("Sorting Time")
ax[1].hist(dataset['log_sorting_time']);ax[1].set_title("Log_Sorting Time")
plt.show()


# In[17]:


sns.jointplot(dataset['delivery_time'],dataset['sorting_time'], kind='kde')
plt.show()


# In[18]:


# square trasformation :
dataset['sqr_delivery_time']= np.square(dataset["delivery_time"])

fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax[0].hist(dataset['delivery_time']);ax[0].set_title("Delivery time")
ax[1].hist(dataset['sqr_delivery_time']);ax[1].set_title("Square_Delivery Time")
plt.show()


# In[19]:


dataset['sqr_sorting_time']= np.square(dataset["sorting_time"])

fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax[0].hist(dataset['sorting_time']);ax[0].set_title("Sorting Time")
ax[1].hist(dataset['sqr_sorting_time']);ax[1].set_title("Square_Sorting Time")
plt.show()


# In[20]:


# square root trasformation :
dataset['sqrt_delivery_time']= np.sqrt(dataset["delivery_time"])

fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax[0].hist(dataset['delivery_time']);ax[0].set_title("Delivery Time")
ax[1].hist(dataset['sqrt_delivery_time']);ax[1].set_title("Sqrt of Delivery Time")
plt.show()


# In[21]:


dataset['sqrt_sorting_time']= np.sqrt(dataset["sorting_time"])

fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax[0].hist(dataset['sorting_time']);ax[0].set_title("sorting Time")
ax[1].hist(dataset['sqrt_sorting_time']);ax[1].set_title("Sqrt of Sorting Time")
plt.show()


# In[22]:


from sklearn.preprocessing import scale
dataset1 = dataset.values
dataset2 = scale(dataset1)  # Used to standardized the dataset
sns.displot(dataset2, kind='kde')
plt.show()


# In[23]:


# Normalize Data set
from sklearn.preprocessing import normalize
dataset2 = normalize(dataset)
plt.hist(dataset2)
plt.show()


# #### Model Building

# In[24]:


model1=smf.ols("delivery_time~sorting_time",data=dataset).fit()


# #### Model Testing

# In[25]:


# Finding Coefficient parameters
model1.params


# In[26]:


# Finding tvalues and pvalues
model1.tvalues , model1.pvalues


# In[27]:


model1.summary()


# In[28]:


# Finding Rsquared Values
model1.rsquared , model1.rsquared_adj


# #### Model Prediction

# In[29]:


# Manual prediction for say sorting time 6
delivery_time = (6.582734) + (1.649020)*(6)
delivery_time


# In[30]:


# Automatic Prediction for say sorting time 6, 8
new_data=pd.Series([6,8])
new_data


# In[31]:


data_pred=pd.DataFrame(new_data,columns=['sorting_time'])
data_pred


# In[32]:


model1.predict(data_pred)


# #### From this model we can understand that above model is the best model.

# # Q2) Build a prediction model for Salary_hike
# ### Build a simple linear regression model by performing EDA and do necessary transformations and select the best model using R or Python.

# In[33]:


# impoort libraries
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf


# In[43]:


df=pd.read_csv(r"C:\Users\Yash Bhosale\Downloads\Salary_Data.csv")
df


# #### EDA and Data Visualization

# In[44]:


df.info()


# In[45]:


df.describe()


# #### Feature Engineering

# In[46]:


df = df.rename(columns={'YearsExperience':'YE','Salary':'Sal'}, inplace=False)
df.head()


# In[48]:


print(skew(df.YE))


# In[49]:


sns.boxplot(df['YE'], orient = 'h')


# In[50]:


# The data for Year Experience is skewed right.
# All the observations lies in the intervals of approximately 3 to 8
# We can say that the median years of experience is 5.2 years.


# In[51]:


sns.distplot(df['YE'])


# In[52]:


# The distribution of Year Experience data is slightly more on the right tail of the curve


# In[53]:


print(skew(df.Sal))


# In[54]:


sns.boxplot(df['Sal'], orient='h')


# In[55]:


# The observations for Salary lies nearly between 57000 to 110000.
# The data is skewed towars right side.
# The median Salary is nearly 65000.


# In[56]:


sns.distplot(df['Sal'])


# In[57]:


sns.pairplot(df)


# In[58]:


sns.distplot(df)


# #### Correlation Analysis

# In[59]:


df.corr()


# In[60]:


sns.heatmap(df.corr(), annot=True)


# #### Transformation for Continuous Variable

# In[61]:


# log trasformation :
df['log_YE']= np.log(df["YE"])

fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax[0].hist(df['YE']);ax[0].set_title("Years Experience")
ax[1].hist(df['log_YE']);ax[1].set_title("Log of Years Experience")
plt.show()


# In[62]:


# log trasformation :
df['log_S']= np.log(df["Sal"])

fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax[0].hist(df['Sal']);ax[0].set_title("Salary")
ax[1].hist(df['log_S']);ax[1].set_title("Log of Salary")
plt.show()


# In[63]:


sns.jointplot(df['YE'],df['Sal'], kind='kde')
plt.show()


# In[64]:


# square trasformation :
df['sqr_YE']= np.square(df["YE"])

fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax[0].hist(df['YE']);ax[0].set_title("Years Experience")
ax[1].hist(df['sqr_YE']);ax[1].set_title("Sqaure of Years Experience")
plt.show()


# In[65]:


# square trasformation :
df['sqr_S']= np.square(df["Sal"])

fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax[0].hist(df['Sal']);ax[0].set_title("Salary")
ax[1].hist(df['sqr_S']);ax[1].set_title("Sqaure of Salary")
plt.show()


# In[66]:


# square root trasformation :
df['sqrt_YE']= np.sqrt(df["YE"])

fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax[0].hist(df['YE']);ax[0].set_title("Years Experience")
ax[1].hist(df['sqrt_YE']);ax[1].set_title("Sqrt of Years Experience")
plt.show()


# In[67]:


# square root trasformation :
df['sqrt_S']= np.sqrt(df["Sal"])

fig, ax=plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax[0].hist(df['Sal']);ax[0].set_title("Salary")
ax[1].hist(df['sqrt_S']);ax[1].set_title("Sqrt of Salary")
plt.show()


# In[68]:


from sklearn.preprocessing import scale
df1 = df.values
df2 = scale(df1)  # Used to standardized the dataset
sns.displot(df2, kind='kde')
plt.show()


# In[69]:


# Normalize Data set
from sklearn.preprocessing import normalize
df2 = normalize(df)
plt.hist(df2)
plt.show()
#df1


# #### Correlation Analysis

# In[70]:


df.corr()


# In[71]:


sns.heatmap(df.corr(), annot=True)


# In[72]:


sns.regplot(x=df['Sal'],y=df['YE'])


# #### Model Building

# In[73]:


model=smf.ols('Sal~YE',data=df).fit()


# #### Model Testing

# In[74]:


model.params


# In[75]:


model.summary()


# In[76]:


#finding tvalues and pvalues
model.tvalues, model.pvalues


# In[77]:


# Finding Rsquared values
model.rsquared , model.rsquared_adj


# #### Model Prediction

# In[78]:


# Manual prediction for say 2 Years Experience
Salary = (25792.200199) + (9449.962321)*(2)
Salary


# In[79]:


# Automatic Prediction for say 4 & 5 Years Experience
new_data=pd.Series([4,5])
new_data


# In[80]:


data_pred=pd.DataFrame(new_data,columns=['YE'])
data_pred


# In[81]:


model.predict(data_pred)


# #### From above we can say that this is the best model.

