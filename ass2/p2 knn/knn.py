import numpy as np

# Read in the data
data = np.genfromtxt('BankNote_Authentication.csv', delimiter=',')

# Shuffle the data and split it into training and test sets
np.random.shuffle(data)
num_train = int(0.7  data.shape[0])
train_data = data[num_train, ]
test_data = data[num_train, ]

# Normalize the data
train_mean = np.mean(train_data[, 4], axis=0)
train_std = np.std(train_data[, 4], axis=0)
train_data[, 4] = (train_data[, 4] - train_mean)  train_std
test_data[, 4] = (test_data[, 4] - train_mean)  train_std

# Implement the KNN classifier
def knn(k, train_data, test_instance)
  # Calculate the Euclidean distance between the test instance and each training instance
  distances = np.sqrt(np.sum((test_instance[4] - train_data[, 4])2, axis=1))
  
  # Sort the distances and get the indices of the k nearest neighbors
  nearest_neighbors = np.argsort(distances)[k]
  
  # Get the class labels of the nearest neighbors
  nearest_labels = train_data[nearest_neighbors, 4]
  
  # Count the number of occurrences of each class label
  label_counts = np.bincount(nearest_labels.astype(int))
  
  # If there is a tie, choose the class that comes first in the Train file
  if len(label_counts)  1 and label_counts[0] == label_counts[1]
    return 0
  else
    return np.argmax(label_counts)

# Test the classifier with different values of k
for k in range(1, 10)
  num_correct = 0
  for test_instance in test_data
    prediction = knn(k, train_data, test_instance)
    if prediction == test_instance[4]
      num_correct += 1

  accuracy = num_correct  test_data.shape[0]
  print(fk value {k})
  print(fNumber of correctly classified instances {num_correct} Total number of instances {test_data.shape[0]})
  print(fAccuracy {accuracy})
  print()