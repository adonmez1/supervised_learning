# Load libraries
import numpy as np
from keras import models
from keras import layers
import data_read as unt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set random seed
np.random.seed(0)

# Set the number of features we want
number_of_features = 23

X = unt.train_load_data()
X_values = X.values
target_i=np.transpose(np.array([X.values[:,10]]))
X_norm=unt.normalize_data(X_values)

X_test = unt.test_load_data()
X_test_values = X_test.values
test_target=np.transpose(np.array([X_test.values[:,10]]))
X_test_norm=unt.normalize_data(X_test_values)

y_train=np.logical_or(X.values[:,3]==46,X.values[:,3]==56)
y_test=np.logical_or(X_test.values[:,3]==46,X_test.values[:,3]==56)

class_weight = {0: 1.,
                1: 2.}

class_weight_2 = {0: 1.,
                  1: 4.}

# Start neural network
network = models.Sequential()

# Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=16, activation='relu', input_shape=(number_of_features,)))

# Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=16, activation='relu'))

# Add fully connected layer with a sigmoid activation function
network.add(layers.Dense(units=1, activation='sigmoid'))

# Compile neural network
network.compile(loss='binary_crossentropy', # Cross-entropy
                optimizer='rmsprop', # Root Mean Square Propagation
                metrics=['accuracy']) # Accuracy performance metric

# Train neural network
history = network.fit(X_norm, # Features
                      y_train, # Target vector
                      epochs=20, # Number of epochs
                      #class_weight=class_weight_2, # To get rid of unbalanced effects, base does not have this
                      verbose=1, # Print description after each epoch
                      batch_size=100, # Number of observations per batch
                      validation_data=(X_test_norm, y_test)) # Data for evaluation

y_pred_2=network.predict(X_test_norm, batch_size=100, verbose=1)
print(confusion_matrix(y_test,y_pred_2.round()))
print(classification_report(y_test,y_pred_2.round()))
print(accuracy_score(y_test, y_pred_2.round()))
