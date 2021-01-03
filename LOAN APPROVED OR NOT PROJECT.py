#!/usr/bin/env python
# coding: utf-8

# Project on loan prediction:Predict if a loan will get approved or not.
# This is a classification problem as we need to
# classify whether the Loan_Status is yes or no.
# IMPLEMENTATION OF PROJECT:
# 

# In[74]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[75]:


df=pd.read_csv("loan prediction.csv")


# In[76]:


##finding the first 5 rows ,shape,description of data


# In[77]:


df.head()


# In[78]:


df.tail()


# In[79]:


df.shape


# In[80]:


df.columns


# In[81]:


df.isnull().sum()


# In[82]:


##we notice so many missing values


# In[83]:


df.describe()


# In[84]:


df.info()


# In[85]:


df['Gender'].value_counts()


# In[86]:


df['Gender']=df['Gender'].fillna('Male')


# In[87]:


df.isnull().sum()


# In[88]:


df['Married'].value_counts()


# In[89]:


df['Married']=df['Married'].fillna('Yes')


# In[90]:


df.isnull().sum()


# In[91]:


df['Self_Employed'].value_counts()


# In[92]:


df['Self_Employed']=df['Self_Employed'].fillna('No')


# In[93]:


df.isnull().sum()


# In[94]:


df['Dependents'].value_counts()


# In[95]:


df['Dependents']=df['Dependents'].fillna('0')


# In[96]:


df.isnull().sum()


# In[97]:


df['LoanAmount'].value_counts()


# In[98]:


mean=df.LoanAmount.mean()
mean


# In[99]:


#df['LoanAmount']=df['LoanAmount'].fillna('mean')


# In[100]:


df.isnull().sum()


# In[101]:


df['Loan_Amount_Term'].value_counts()


# In[102]:


df.LoanAmount=df.LoanAmount.fillna(mean)


# In[103]:


df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna('360.0')


# In[104]:


df.isnull().sum()


# In[105]:


df['Credit_History'].value_counts()


# In[106]:


df['Credit_History']=df['Credit_History'].fillna('1.0')


# In[107]:


df.isnull().sum()


# In[108]:


from sklearn.preprocessing import LabelEncoder
var_mod =['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
for i in var_mod:
 df[i] = le.fit_transform(df[i])


# In[109]:


df.head()


# In[110]:


X = df.iloc[:,1:12]
y = df.iloc[:,12]


# In[111]:


from sklearn.model_selection import train_test_split


# In[112]:


X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.20,random_state=0)


# In[113]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=0)


# In[114]:


from sklearn.tree import DecisionTreeClassifier


# In[115]:


model= DecisionTreeClassifier() 


# In[116]:


model.fit(X_train,y_train) 


# In[117]:


y_predictions= model.predict(X_test)


# In[127]:


from sklearn.metrics import accuracy_score,classification_report


# In[137]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test,y_pred)
print(cnf_matrix)


# from sklearn import metrics

# In[130]:


print("Accuracy:",metrics.accuracy_score(y_test,y_predictions))


# In[131]:


from sklearn.linear_model import LogisticRegression


# In[132]:


logistic_regression= LogisticRegression() 


# In[134]:


logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)


# In[135]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test,y_pred)
print(cnf_matrix)


# In[136]:


print("Accuracy:",metrics.accuracy_score(y_test,y_pred))


# Let's predict whether the loan will get approved or
# not for a person(John) who is applying for the
# loan with the following details:
# Gender : Male
# Married: Yes
# Dependents: 1
# Education: Graduate
# Self_Employed: No
# ApplicantIncome: 8000
# CoapplicantIncome: 2000
# LoanAmount (in thousand): 130
# Loan_Amount_Term(Term of loan in months): 24
# Credit_History: 0.0
# Property_Area (Urban/ Semi Urban/ Rural): Urban
# 

# In[141]:


Loanstatus =logistic_regression.predict([[1,1,1,0,0,8000,2000,130,24,0.0,2]])
print(Loanstatus)


# In[ ]:


##This means that the loan get approved for the
person (John) and he will get the loan.###

