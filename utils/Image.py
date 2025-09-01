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