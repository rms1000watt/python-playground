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
      labels.extend(out["labels"])
      filenames.extend(out["filenames"])

def main():
  loadCIFAR10()
  
  # Probability of dropout
  p = 0.5 

  # len(d0) == 3072 == 3 * 1024
  d0 = data[0]

  # Get Bias
  b = np.mean(d0, axis=0)

  # Zero center data
  d0 = np.subtract(d0, b)

  # Normalize data
  # d0 = np.divide(d0, np.std(d0, axis=0))

  # Create small, random weights. len(d0) == 3072
  W1 = np.random.randn(10, len(d0)) * np.sqrt(2./len(d0))

  # Multiply by weights
  H = np.matmul(W1, d0)

  # Rectify
  H = np.maximum(0, H)

  # Create mask (inverted dropout)
  print(*H.shape)
  M = (np.random.rand(*H.shape) < p) / p
  print(H)





if __name__ == '__main__':
  main()