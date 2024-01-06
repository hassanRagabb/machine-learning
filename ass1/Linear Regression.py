#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


cars = pd.read_csv("car_data.csv")
cars.info()


# In[3]:


#creating df  with numeric var's only
cars_numeric=cars.select_dtypes(include=['float64','int64'])
# dropping symboling and car_ID as symboling is more of categorical variable as described before and car_ID is only 
#an index type variable and not a predictor
cars_numeric = cars_numeric.drop(['symboling', 'ID'], axis=1)
cars_numeric.head()


# In[5]:


import seaborn as sns
for i, col in enumerate (cars_numeric.columns):
    plt.figure(i)
    sns.scatterplot(x=cars_numeric[col],y=cars_numeric['price'])#x is feature ,y is price
    
#These var's appears to have a linear relation with price: carwidth, curbweight, enginesize, horsepower, boreration and citympg.


# In[6]:


corr=cars_numeric.corr()

plt.figure(figsize=(15,8))
sns.heatmap(corr,annot=True,cmap="YlGnBu")
#Positive corr: Price highly correlated with enginesize, curbweight, horsepower, carwidth


# In[59]:


car=cars.drop(columns=['symboling','fueltypes','aspiration','doornumbers','carbody','drivewheels',
                     'enginelocation','wheelbase','carheight','enginetype','cylindernumber','fuelsystem','boreratio'
                     ,'stroke','compressionratio','peakrpm','citympg','highwaympg','carlength','name',"ID"])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
car=scaler.fit_transform(car)
X = car[:, :-1]
y = car[:, -1]
# X=car.drop(columns=['price'])
# y=car['price']
# print(y)
# split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,test_size = 0.3, random_state=100)

# Model with  features
# from sklearn import linear_model
# from sklearn.linear_model import LinearRegression

# lm=LinearRegression()
# lm.fit(X_train,y_train)

# y_pred_test=lm.predict(X_test)
# y_pred_train=lm.predict(X_train)

# #Rsqaure
# from sklearn.metrics import r2_score

# print('R-sqaure on train data: {}'.format(r2_score(y_true=y_train, y_pred=y_pred_train)))
# print('R-sqaure on test data: {}'.format(r2_score(y_true=y_test, y_pred=y_pred_test)))


# In[105]:


class LinearRegression:

    def __init__(self, lr = 0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
            

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    
    def mse(y_test, predictions):
        return np.mean((y_test-predictions)**2)

    
from sklearn import datasets
import matplotlib.pyplot as plt


X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,test_size = 0.3, random_state=100)

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
plt.show()

reg = LinearRegression(lr=0.01)
reg.fit(X_train,y_train)
predictions = reg.predict(X_test)

def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)
#----------
msels= []
iterationls= []
for i in range(100):
    
    reg = LinearRegression(lr=0.05, n_iters=i)
    reg.fit(X_train,y_train)
    predictions = reg.predict(X_test)
    msels.append(mse(y_test, predictions))
    iterationls.append(i)
    print(msels[i])
    
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

# x = np.array([5, 4, 1, 4, 5])
# y = np.sort(x)

plt.title("Line graph")
plt.plot(iterationls, msels, color="red")

plt.show()


# mse = mse(y_test, predictions)
# print(mse)

y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()


# In[ ]:





# In[ ]:




