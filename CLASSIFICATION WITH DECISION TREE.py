#!/usr/bin/env python
# coding: utf-8

# # Importing libraries
# 

# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")


# # Import the dataset
# 

# In[12]:


df = pd.read_csv("C:/Users/ak320/Downloads/archive (4)/Iris.csv")
df


# # Preprocess the data

# In[13]:


df.head()


# In[14]:


df.tail()


# In[16]:


df.describe(include = 'all')


# In[18]:


df.describe(include = 'O')


# In[19]:


df.columns


# In[20]:


df['Species'].unique()


# In[21]:


df['Species'].value_counts()


# In[22]:


# Plot for species column by count
sns.countplot(df['Species'], palette = 'viridis')


# # Pairplot of the dataset

# In[48]:


# pairplot

sns.pairplot(df, hue = 'Species', diag_kind = 'hist',markers=["o", "s", "D"])


# # Correlation matrix of the dataset

# In[24]:


# checking the correlation matrix of the categorical variables
df.corr()


# # Heatmap of the correlation matrix

# In[25]:


sns.heatmap(df.corr(),cmap = 'crest', annot = True)


# # Encoding categorical variables using LabelEncoder

# In[26]:


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()


# In[27]:


df.drop('Id',axis = 1,inplace = True)
df.iloc[:,-1] = LE.fit_transform(df.iloc[:,-1])
df


# # Splitting the dataset into training and testing sets

# In[29]:


x=df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y=df.iloc[:,-1] 


# In[30]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 50)


# In[31]:


x_train.head()


# # Checking the shapes of the training and testing sets
# 
# 

# In[32]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# #  Training a Decision Tree Classifier

# In[33]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()


# In[34]:


dt.fit(x_train,y_train)


# #  Making predictions on the testing set

# In[35]:


y_pred = dt.predict(x_test)
y_pred


# In[36]:


y_test= np.array(y_test)
y_test


# In[37]:


final_pred = pd.DataFrame( { 'Actual':  y_test,
                            'Predicted': dt.predict( x_test) } )


# In[38]:


final_pred.sample(n=12)


# #  Calculating the accuracy score

# In[39]:


from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)


# # Generating a classification report

# In[40]:


from sklearn.metrics import classification_report
print(classification_report(y_pred,y_test))


# In[41]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_pred,y_test)


# #  Confusion Matrix

# In[42]:


from mlxtend.plotting import plot_confusion_matrix


# # Plotting Confusion Matrix 

# In[43]:


plot_confusion_matrix(confusion_matrix(y_pred,y_test))


# # Visualizing Decision Tree

# In[44]:


from sklearn import tree


# In[45]:


plt.figure(figsize = (20,15))
decison_tree = tree.plot_tree(dt, feature_names = x.columns, filled = True, fontsize = 20)


# In[ ]:




