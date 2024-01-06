#!/usr/bin/env python
# coding: utf-8

# In[167]:


# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# load dataset
pima = pd.read_csv("BankNote_Authentication.csv")
pima.head()


# In[168]:


feature_cols = ['variance', 'skewness', 'curtosis', 'entropy']
X = pima[feature_cols]
y = pima['class']


# In[173]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=1)
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('BankNote_Authentication.png')
Image(graph.create_png())


# In[176]:


import matplotlib.pyplot as plt
  
# train size
x = [30,40,50,60,70]
# accuracy mean
y = [0.98231009,0.982767,0.983965,0.986714,0.986893]
  
# plotting the points 
plt.plot(x, y)
  
plt.xlabel('train size %')
plt.ylabel('accuracy mean')
plt.title('accuracy to train size')
  
# function to show the plot
plt.show()


# In[177]:


# train size
x = [30,40,50,60,70]

# Number of nodes
y = [31,37,33,35,51]
  
# plotting the points 
plt.plot(x, y)
  
plt.xlabel('train size %')
plt.ylabel('tree size')
plt.title('tree size to train size')
  
# function to show the plot
plt.show()


# In[ ]:




