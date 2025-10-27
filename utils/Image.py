import glob, imageio, matplotlib.pyplot as plt

def ShowImage(path: int):
  image = plt.imread(path)
  plt.imshow(image)
  plt.show()

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