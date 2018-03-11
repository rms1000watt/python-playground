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
  
  # len(data[0]) == 3072
  W1 = np.random.rand(10,3072)
  print(data[0])
  out = np.matmul(W1, data[0])
  print(out)
  out = np.maximum(0, out)
  print(out)





if __name__ == '__main__':
  main()