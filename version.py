import sys
import tensorflow as tf
import importlib.metadata as md

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)

try:
    print("tf-keras:", md.version("tf-keras"))
except md.PackageNotFoundError:
    print("tf-keras: not installed")