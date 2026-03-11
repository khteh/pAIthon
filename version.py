import sys, tensorflow as tf, importlib.metadata as md

print(f"Python: {sys.version}")
print(f"TensorFlow: {tf.__version__} {tf.version.VERSION}")

try:
    print(f"tf-keras: {md.version("tf-keras")}")
except md.PackageNotFoundError:
    print("tf-keras: not installed")