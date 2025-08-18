import glob, imageio, matplotlib.pyplot as plt, os, PIL, time
import numpy, math, tensorflow as tf
"""
https://www.tensorflow.org/tutorials/generative/dcgan
This "common" function is only tested working with MNISTGAN.
"""
def restore_latest_checkpoint(checkpoint, checkpoint_dir):
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def show_image(path: int):
    # Display a single image using the epoch number
  return PIL.Image.open(path)

def CreateGIF(anim_file: str, input_images: str):
    """
    Use imageio to create an animated gif using the images saved during training.
    """
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(input_images)
        filenames = sorted(filenames)
        for f in filenames:
            image = imageio.imread(f)
            writer.append_data(image)
    print(f"{anim_file} created successfully!")