import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import scipy
from PIL import Image
import numpy as np
import h5py
import numpy as np
import copy
# from public_tests import *

# Load training data
train_dataset = h5py.File("train_catvsnoncat.h5", "r")
train_set_x = np.array(train_dataset["train_set_x"][:])  # images
train_set_y = np.array(train_dataset["train_set_y"][:])  # labels

# Load test data
test_dataset = h5py.File("test_catvsnoncat.h5", "r")
test_set_x = np.array(test_dataset["test_set_x"][:])
test_set_y = np.array(test_dataset["test_set_y"][:])

# Classes (0 = non-cat, 1 = cat)
classes = np.array(test_dataset["list_classes"][:])

print("Training data shape:", train_set_x.shape)
print("Training labels shape:", train_set_y.shape)
print("Test data shape:", test_set_x.shape)
print("Test labels shape:", test_set_y.shape)
print("Classes:", classes)

train_x = train_set_x.reshape(train_set_x.shape[0], -1).T
test_x = test_set_x.reshape(test_set_x.shape[0], -1).T

print(train_x.shape)
print(test_x.shape)

train_x = train_x/255
test_x = test_x/255


def sigmoid(z) :
    s = 1/ (1+ np.exp(-z))
    return s

def propagate(w,b,X,Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X) + b)
    cost = -(1/m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
    
    dw = (1/m) * np.dot(X,(A-Y).T)
    db = (1/m) * np.sum(A-Y)
    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw,
             "db": db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []
    
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(float(cost))
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T,X) + b)
    for i in range(A.shape[1]):
        if A[0,i] > 0.5 :
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w , b  = np.zeros(shape = (train_x.shape[0],1)) , 0.0
    params, grads, costs = optimize(w,b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]
    
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)
    
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
logistic_regression_model = model(train_x, train_set_y, test_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)


# Accuracy
train_accuracy = 100 - np.mean(np.abs(logistic_regression_model["Y_prediction_train"] - train_set_y)) * 100
test_accuracy  = 100 - np.mean(np.abs(logistic_regression_model["Y_prediction_test"]  - test_set_y)) * 100
print(f"\nTrain Accuracy: {train_accuracy:.2f}%")
print(f"Test  Accuracy: {test_accuracy:.2f}%")

# plt.plot(np.squeeze(logistic_regression_model["costs"]))
# plt.ylabel('cost'); plt.xlabel('iterations (per hundreds)')
# plt.title(f"Learning rate = {logistic_regression_model['learning_rate']}")
# plt.show()

import pickle

# Save model
with open("cat_model.pkl", "wb") as f:
    pickle.dump(logistic_regression_model, f)

print("✅ Model saved as cat_model.pkl")

print(classes)
