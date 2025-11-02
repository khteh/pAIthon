import glob, matplotlib.pyplot as plt, imageio, imageio.v3 as iio
from .TermColour import bcolors

def ShowImage(path: int):
  image = iio.imread(path)
  plt.figure(figsize=(10, 10))
  plt.tight_layout(pad=0.1,rect=[0, 0, 1, 0.98]) #[left, bottom, right, top]
  plt.axis('off')
  plt.imshow(image)
  plt.show()

def CreateGIF(gif_path: str, input_images: str, duration:float = 0.2):
    """
    Use imageio to create an animated gif using the images saved during training.
    """
    filenames = glob.glob(input_images)
    filenames = sorted(filenames)
    images = [iio.imread(f) for f in filenames]
    # Save the frames as a GIF that plays once (loop=1)
    imageio.mimsave(gif_path, images, duration=duration, loop=1)
    print(f"{bcolors.OKGREEN}{gif_path} created successfully!{bcolors.DEFAULT}")

# Assuming 'images' is a TensorFlow tensor of shape (batch_size, H, W, C)
def make_image_grid(images, num_rows, num_cols):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    axes = axes.flatten()
    for i, img in enumerate(images):
        if i < len(axes):
            axes[i].imshow(img.numpy()) # .numpy() to convert tensor to numpy array for matplotlib
            axes[i].axis('off')
    plt.tight_layout()
    plt.show()