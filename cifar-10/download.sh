#!/usr/bin/env bash

CIFAR10_URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_TGZ="cifar-10-python.tar.gz"
CIFAR10_DIR="cifar-10-batches-py"

if [ ! -f "$CIFAR10_TGZ" ] && [ ! -d "$CIFAR10_DIR" ]; then
  curl -o "$CIFAR10_TGZ" "$CIFAR10_URL"
fi

if [ -f "$CIFAR10_TGZ" ] && [ ! -d "$CIFAR10_DIR" ]; then
  tar -zxvf "$CIFAR10_TGZ"
fi

