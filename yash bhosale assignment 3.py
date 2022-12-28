#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# # Assignment - 03 (Hypothesis Testing)

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import chi2_contingency


# ###### Q1. A F&B manager wants to determine whether there is any significant difference in the diameter of the cutlet between two units. A randomly selected sample of cutlets was collected from both units and measured? Analyze the data and draw inferences at 5% significance level. Please state the assumptions and tests that you carried out to check validity of the assumptions.

# In[2]:


# Load the dataset
data=pd.read_csv(r"C:\Users\Yash Bhosale\Downloads\Cutlets (2).csv")
data.head()


# In[3]:


# Assume Null hyposthesis as Ho: μ1 = μ2 (There is no difference in diameters of cutlets between two units)
# Thus Alternate hypothesis as Ha: μ1 ≠ μ2 (There is significant difference in diameters of cutlets between two units)
#2 Sample 2 Tail test applicable


# In[4]:


unitA=pd.Series(data.iloc[:,0])
unitA


# In[5]:


unitB=pd.Series(data.iloc[:,1])
unitB


# In[6]:


# 2-sample 2-tail ttest:   stats.ttest_ind(array1,array2)     # ind -> independent samples
p_value=stats.ttest_ind(unitA,unitB)
p_value


# In[7]:


# 2-tail probability 
p_value[1]     


# In[18]:


# compare p_value with α = 0.05 (At 5% significance level)


# In[19]:


# Inference:
# As (p_value=0.4722) > (α = 0.05); Accept Null Hypothesis
# i.e. μ1 = μ2 
# Thus, there is no difference in diameters of cutlets between two units


# ###### Q2. A hospital wants to determine whether there is any difference in the average Turn Around Time (TAT) of reports of the laboratories on their preferred list. They collected a random sample and recorded TAT for reports of 4 laboratories. TAT is defined as sample collected to report dispatch. Analyze the data and determine whether there is any difference in average TAT among the different laboratories at 5% significance level.

# In[20]:


# load the dataset
data=pd.read_csv(r"C:\Users\Yash Bhosale\Downloads\LabTAT (3).csv")
data.head()


# In[23]:


# Anova ftest statistics: Analysis of varaince between more than 2 samples or columns
# Assume Null Hypothesis Ho as No Varaince: All samples TAT population means are same
# Thus Alternate Hypothesis Ha as It has Variance: Atleast one sample TAT population mean is different 


# In[21]:


# Anova ftest statistics: stats.f_oneway(column-1,column-2,column-3,column-4)
p_value=stats.f_oneway(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2],data.iloc[:,3])
p_value


# In[22]:


# compare it with α = 0.05
p_value[1]  


# In[24]:


# Inference:
# As (p-value=0) < (α = 0.05); Reject Null Hypothesis
# i.e. Atleast one sample TAT population mean is different 
# Thus there is variance or difference in average Turn Around Time (TAT) of reports of the laboratories on their preferred list.


# ###### Q.3

#  ![Screenshot%20%2818%29.png](attachment:Screenshot%20%2818%29.png)

# In[ ]:


# Assume Null Hypothesis as Ho: 
# Independence of categorical variables (male-female buyer rations are similar across regions (does not vary and are not related)

# Thus Alternate Hypothesis as Ha: 
# Dependence of categorical variables (male-female buyer rations are NOT similar across regions (does vary and somewhat/significantly related)


# In[13]:


# load the dataset
data=pd.read_csv(r"C:\Users\Yash Bhosale\Downloads\BuyerRatio (2).csv")
data


# In[14]:


# Make dimensional array
obs=np.array([[50,142,131,70],[435,1523,1356,750]])
obs


# In[15]:


# Chi2 contengency independence test
chi2_contingency(obs) # o/p is (Chi2 stats value, p_value, df, expected obsvations)


# In[25]:


# Compare p_value with α = 0.05


# In[26]:


# Inference:
# As (p-value = 0.6603) > (α = 0.05); Accept the Null Hypothesis
# i.e. Independence of categorical variables 
# Thus, male-female buyer rations are similar across regions and are not related


# ###### Q4. TeleCall uses 4 centers around the globe to process customer order forms. They audit a certain %  of the customer order forms. Any error in order form renders it defective and has to be reworked before processing.  The manager wants to check whether the defective %  varies by centre. Please analyze the data at 5% significance level and help the manager draw appropriate inferences.

# In[ ]:


# load the dataset
data=pd.read_csv(r"C:\Users\Yash Bhosale\Downloads\Costomer+OrderForm (2).csv")
data


# In[ ]:


data.Phillippines.value_counts()


# In[ ]:


data.Indonesia.value_counts()


# In[ ]:


data.Malta.value_counts()


# In[ ]:


data.India.value_counts()


# In[ ]:


# Make a contingency table
obs=np.array([[271,267,269,280],[29,33,31,20]])
obs


# In[ ]:


# Assume Null Hypothesis as Ho: 
# Independence of categorical variables (customer order forms defective %  does not varies by centre)

# Thus, Alternative hypothesis as Ha
# Dependence of categorical variables (customer order forms defective %  varies by centre)


# In[ ]:


# Chi2 contengency independence test
chi2_contingency(obs) # o/p is (Chi2 stats value, p_value, df, expected obsvations)


# In[ ]:


# Compare p_value with α = 0.05


# In[ ]:


# Inference:
# As (p_value = 0.2771) > (α = 0.05); Accept Null Hypthesis
# i.e. Independence of categorical variables 
# Thus, customer order forms defective %  does not varies by centre


# In[ ]:

