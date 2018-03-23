#!/usr/bin/env python3

import numpy as np
import _pickle as pickle

baseFilename = "cifar-10-batches-py/data_batch_"
data = []
labels = []
filenames = []

def loadCIFAR10():
  for i in range(1,2):
    with open(baseFilename + str(i), "rb") as f:
      out = pickle.load(f, encoding="latin1")
      data.extend(out["data"])
      filenames.extend(out["filenames"])
      global labels
      for label in out["labels"]:
        z = [0]*10
        z[label] = 1
        labels.append(z)

def sig(x): return 1/(1+np.exp(-x))
def dSig(x): return x*(1-x)

def main():
  loadCIFAR10()

  # Output Labels (50, 10)
  y = labels[0:50]

  # Input Images (50, 3072)
  x = np.array(data[0:50])

  # Center on 0
  x = np.subtract(x, 128.)

  # Normalize data
  x = np.divide(x, 128.)

  # Create small, random weights
  W1 = (2.*(np.random.rand(3072, 10)) - 1.) * 1e-2
  
  for i in range(10000):
    # Multiply by weights
    H1 = np.matmul(x, W1)

    # Rectify
    # H1 = np.maximum(0, H1)
    H1 = sig(H1)

    # Create mask (inverted dropout)
    # M = (np.random.rand(*H1.shape) < p) / p
    # H1 = np.multiply(H1, M)

    # Find Error
    err = np.subtract(y, H1)

    # Calculate delta
    delta = err * dSig(H1) * 1e-2

    W1 += np.dot(x.T, delta)
  
    if i%100 is 0:
      print(i, ":", H1[0])

  np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
  print("final:", H1[0])
  print("actual:", y[0])

  print("final:", H1[1])
  print("actual:", y[1])

  print("final:", H1[2])
  print("actual:", y[2])

  print("W1:", W1)


if __name__ == '__main__':
  main()

