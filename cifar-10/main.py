#!/usr/bin/env python3

import numpy as np
import _pickle as pickle

baseFilename = "cifar-10-batches-py/data_batch_"
data = []
labels = []
filenames = []

# Probability of dropout
p = 0.5

def loadCIFAR10():
  for i in range(1,2):
    with open(baseFilename + str(i), "rb") as f:
      out = pickle.load(f, encoding="latin1")
      print(out.keys())
      data.extend(out["data"])
      # labels.extend(out["labels"])
      filenames.extend(out["filenames"])
      global labels
      for label in out["labels"]:
        z = [0]*10
        z[label] = 1
        labels.append(z)

def main():
  loadCIFAR10()
  print(data[0:3])
  print(labels[0:3])
  print(filenames[0:3])

  print(len(labels))
  print(labels[0])
  exit()

  # (3072, 50)
  d = np.mat(data[0:50]).T
  
  # Get Bias
  b = np.mean(d, axis=0)

  # Zero center data
  d = np.subtract(d, b)

  # Normalize data
  # d = np.divide(d, np.std(d, axis=0))

  # Create small, random weights
  W1 = np.random.randn(10, 3072) * np.sqrt(2./len(d))

  # Multiply by weights
  H1 = np.matmul(W1, d)

  # Rectify
  H1 = np.maximum(0, H1)

  # Create mask (inverted dropout)
  M = (np.random.rand(*H1.shape) < p) / p
  H1 = np.multiply(H1, M)
  print(H1)
  


if __name__ == '__main__':
  main()


# def nonlin(x,deriv=False):
#     if(deriv==True):
#         return x*(1-x)
#     return 1/(1+np.exp(-x))
    
# # input dataset
# X = np.array([  [0,0,1],
#                 [0,1,1],
#                 [1,0,1],
#                 [1,1,1] ])
    
# # output dataset            
# y = np.array([[0,0,1,1]]).T

# # seed random numbers to make calculation
# # deterministic (just a good practice)
# np.random.seed(1)

# # initialize weights randomly with mean 0
# syn0 = 2*np.random.random((3,1)) - 1

# for iter in range(20000):

#     # forward propagation
#     l0 = X
#     l1 = nonlin(np.dot(l0,syn0))

#     # how much did we miss?
#     l1_error = y - l1

#     # multiply how much we missed by the 
#     # slope of the sigmoid at the values in l1
#     l1_delta = l1_error * nonlin(l1,True)

#     # update weights
#     syn0 += np.dot(l0.T,l1_delta)

# print("Output After Training:")
# print(l1)

#Input array
X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])

#Output
y=np.array([[1],[1],[0]])

#Sigmoid Function
def sigmoid (x):
  return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
  return x * (1 - x)

#Variable initialization
epoch=20000 #Setting training iterations
lr=0.1 #Setting learning rate
inputlayer_neurons = X.shape[1] #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer

#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))

for i in range(epoch):

  #Forward Propogation
  hidden_layer_input1=np.dot(X,wh)
  hidden_layer_input=hidden_layer_input1 + bh
  hiddenlayer_activations = sigmoid(hidden_layer_input)
  output_layer_input1=np.dot(hiddenlayer_activations,wout)
  output_layer_input= output_layer_input1+ bout
  output = sigmoid(output_layer_input)

  #Backpropagation
  E = y-output
  slope_output_layer = derivatives_sigmoid(output)
  slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
  d_output = E * slope_output_layer
  Error_at_hidden_layer = d_output.dot(wout.T)
  d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
  wout += hiddenlayer_activations.T.dot(d_output) *lr
  bout += np.sum(d_output, axis=0,keepdims=True) *lr
  wh += X.T.dot(d_hiddenlayer) *lr
  bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr

print(output)
