#!/usr/bin/env python3

from abc import ABCMeta, abstractmethod

# Interfaces
class Barker(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def bark(self):
        pass

class Swimmer(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def swim(self):
        pass

class Walker(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def walk(self):
        pass

# Implementations of Interfaces above
class Dog(Barker, Walker):
  def __init__(self, name):
    self.name = name

  def bark(self):
    print("bark")

  def walk(self):
    print("walking...")
  
  def sit(self):
    print("sitting...")

class Seal(Barker, Swimmer):
  def __init__(self):
    pass
  
  def bark(self):
    print("bark bark")
  
  def swim(self):
    print("swimming...")


dog = Dog("Fido")
seal = Seal()
