#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import itertools

class LogitRegression() :
    def __init__( self, learning_rate, iterations ) :        
        self.learning_rate = learning_rate        
        self.iterations = iterations
          
    # Function for model training    
    def fit( self, X, Y ) :        
        # no_of_training_examples, no_of_features        
        self.m, self.n = X.shape        
        # weight initialization        
        self.W = np.zeros( self.n )        
        self.b = 0        
        self.X = X        
        self.Y = Y
          
        # gradient descent learning
                  
        for i in range( self.iterations ) :            
            self.update_weights()            
        return self
      
    # Helper function to update weights in gradient descent
      
    def update_weights( self ) :           
        A = 1 / ( 1 + np.exp( - ( self.X.dot( self.W ) + self.b ) ) )
        # calculate gradients        
        tmp = ( A - self.Y.T )        
        tmp = np.reshape( tmp, self.m )        
        dW = np.dot( self.X.T, tmp ) / self.m         
        db = np.sum( tmp ) / self.m 
          
        # update weights    
        self.W = self.W - self.learning_rate * dW    
        self.b = self.b - self.learning_rate * db
          
        return self
      
    # Hypothetical function  h( x ) 
      
    def predict( self, X ) :    
        Z = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) )        
        Y = np.where( Z > 0.5, 1, 0 )        
        return Y
  
  
# Driver code
  
def main() :

    # Importing dataset with pandas 
    df = pd.read_csv( "customer_data.csv" )
    XNotScaled  = df.iloc[:,:-1].values
    Y = df.iloc[:,-1:].values
    
    # Importing dataset with readline 
    f = open("customer_data.csv")
    f.readline()  # skip the header
    data =  np.loadtxt(fname = f, delimiter = ',')
    #XNotScaled = data[:, 0:2]
    #Y = data[:, 2:]
    
    #scaling the features
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(XNotScaled)
    
    
    # Splitting dataset into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(
      X, Y, test_size = 2/10, random_state = 1 )
    
    # Model training    
    model = LogitRegression( learning_rate = 0.52, iterations = 1000 )
      
    model.fit( X_train, Y_train )    
      
    # Prediction on test set
    Y_pred = model.predict( X_test )    

      
    # measure performance    
    correctly_classified = 0    

    
    # counter    
    count = 0    
    for count in range( np.size( Y_pred ) ) :  
        
        if Y_test[count] == Y_pred[count] :            
            correctly_classified = correctly_classified + 1
          

        count = count + 1
          
    print( "Accuracy :  ", ( correctly_classified / count ) * 100 )
      
    print(Y_pred)
    print(list(itertools.chain.from_iterable(Y_test)))
    
if __name__ == "__main__" :     
    main()


# In[4]:


X.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




