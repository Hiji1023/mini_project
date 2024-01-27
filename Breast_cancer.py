#!/usr/bin/env python
# coding: utf-8

# # CASE STUDY: BREAST CANCER CLASSIFICATION
# # Dr. Rayan Ahmed

# # STEP 1: PROBLEM STATEMENT

# 
# - Predicting if the cancer diagnosis is benign or malignant based on several observations/features 
# - 30 features are used, examples:
#         - radius (mean of distances from center to points on the perimeter)
#         - texture (standard deviation of gray-scale values)
#         - perimeter
#         - area
#         - smoothness (local variation in radius lengths)
#         - compactness (perimeter^2 / area - 1.0)
#         - concavity (severity of concave portions of the contour)
#         - concave points (number of concave portions of the contour)
#         - symmetry 
#         - fractal dimension ("coastline approximation" - 1)
# 
# - Datasets are linearly separable using all 30 input features
# - Number of Instances: 569
# - Class Distribution: 212 Malignant, 357 Benign
# - Target class:
#          - Malignant
#          - Benign
# 
# 
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
# 
# ![image.png](attachment:image.png)

# # STEP 2: IMPORTING DATA

# In[65]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[66]:


from sklearn.datasets import load_breast_cancer


# In[67]:


cancer = load_breast_cancer()


# In[68]:


cancer


# In[69]:


cancer.keys()


# In[70]:


print(cancer ['DESCR'])


# In[71]:


print(cancer ['target'])


# In[72]:


print(cancer ['target_names'])


# In[73]:


print(cancer ['feature_names'])


# In[74]:


cancer['data'].shape
# 30 raw =  30 features


# In[75]:


# creates data frame using pandas
df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']], columns = np.append(cancer['feature_names'],['target']))


# In[76]:


df_cancer.head()


# In[77]:


df_cancer.tail()


# # STEP 3: VISUALING THE DATA

# In[78]:


sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter',
 'mean smoothness'])


# In[79]:


sns.countplot(df_cancer['target'], label ="Count")


# In[80]:


sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)


# In[81]:


#correlation

plt.figure(figsize=(20,10)) 
sns.heatmap(df_cancer.corr(), annot=True) 


# # STEP 4: MODEL TRAINING (FINDING A PROBLEM SOLUTION)

# In[82]:


X = df_cancer.drop(['target'], axis = 1)


# In[83]:


X


# In[84]:


y = df_cancer['target']
y


# In[85]:


# split data into train data, test data
from sklearn.model_selection import train_test_split 


# In[86]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


# In[87]:


X_train.shape


# In[88]:


y_train.shape


# In[89]:


X_test.shape


# In[90]:


y_test.shape


# In[91]:


from sklearn.svm import SVC


# In[92]:


from sklearn.metrics import classification_report, confusion_matrix


# In[93]:


svc_model = SVC()


# In[94]:


svc_model.fit(X_train, y_train)
svc_model.get_params()


# # STEP 5:  EVALUATING THE MODEL

# In[95]:


y_predict = svc_model.predict(X_test)


# In[96]:


y_predict


# In[97]:


cm = confusion_matrix(y_test, y_predict)


# In[98]:


sns.heatmap(cm, annot=True)


# In[99]:


print(classification_report(y_test,y_predict))


# # STEP 6: IMPROVING THE MODEL

# In[100]:


# Normalization
#Feature Scaling
# 정규화 수행

min_train = X_train.min()
min_train


# In[101]:


range_train = (X_train - min_train).max()
range_train


# In[102]:


X_train_scaled = (X_train - min_train)/range_train


# In[103]:


X_train_scaled


# In[104]:


sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)


# In[105]:


sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)


# In[106]:


min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test


# In[107]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC(gamma = 'auto')
svc_model.fit(X_train_scaled, y_train)
svc_model.get_params()


# In[108]:


y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm, annot=True, fmt='d')


# In[109]:


print(classification_report(y_test, y_predict))

# normalization --


# # IMPROVING THE MODEL - PART 2

# In[139]:


param_grid = {'C' : [0.1, 1, 10, 100], 'gamma' : [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}


# In[140]:


from sklearn.model_selection import GridSearchCV


# In[141]:


grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=4)


# In[142]:


grid.fit(X_train_scaled, y_train)


# In[143]:


grid.best_params_


# In[144]:


grid.best_estimator_


# In[146]:


grid_predictions = grid.predict(X_test_scaled)


# In[148]:


cm = confusion_matrix(y_test, grid_predictions)


# In[149]:


sns.heatmap(cm, annot=True)


# In[151]:


print(classification_report(y_test, grid_predictions))


# In[ ]:




