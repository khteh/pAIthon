import glob, imageio, matplotlib.pyplot as plt, os, PIL, time
import numpy, math, tensorflow as tf

def restore_latest_checkpoint(checkpoint, checkpoint_dir):
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def TrainStep(images, discriminator, generator, batch_size: int):
    noise_dim = 100
    """
    The training loop begins with generator receiving a random seed as input. That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). 
    The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.
    """
    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator.run(noise, training=True)

      real_output = discriminator.run(images, training=True)
      fake_output = discriminator.run(generated_images, training=True)

      gen_loss = generator.loss(fake_output)
      disc_loss = discriminator.loss(real_output, fake_output)

    generator.UpdateParameters(gen_tape, gen_loss)
    discriminator.UpdateParameters(disc_tape, disc_loss)

def Train(dataset, epochs: int, discriminator, generator, checkpoint_path, batch_size: int, num_examples_to_generate: int, image_rows: int, image_cols: int):
    checkpoint = tf.train.Checkpoint(generator_optimizer = generator.optimizer,
                                    discriminator_optimizer = discriminator.optimizer,
                                    generator = generator,
                                    discriminator = discriminator)
    noise_dim = 100
    #num_examples_to_generate = 16
    # Reuse this seed overtime so that it's easier to visualize progress in the animated GIF
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    """
    The training loop begins with generator receiving a random seed as input. That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). 
    The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.
    """
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            TrainStep(image_batch, discriminator, generator, batch_size)

        # Produce images for the GIF as you go
        save_images(generator.run(seed, training=False), f'image_at_epoch_{epoch+1:04d}.png', (image_rows, image_cols))

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_path)

        print(f"Time for epoch {epoch + 1} is {time.time()-start}s")

    # Generate after the final epoch
    save_images(generator.run(seed, training=False), f'image_at_epoch_{epoch:04d}.png', (image_rows, image_cols))
    return checkpoint

def save_images(data, filename: str, dimension):
    # Notice `training` is set to False. This is so all layers run in inference mode (batchnorm).
    #fig = plt.figure(figsize=(4, 4))
    fig = plt.figure(figsize=dimension)
    for i in range(data.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(data[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig(filename)
    plt.show()

def show_image(epoch: int):
    # Display a single image using the epoch number
  return PIL.Image.open(f'image_at_epoch_{epoch:04d}.png')

def CreateGIF(filename: str):
    """
    Use imageio to create an animated gif using the images saved during training.
    """
    with imageio.get_writer(filename, mode='I') as writer:
        filenames = glob.glob('image*.png')
        filenames = sorted(filenames)
        for f in filenames:
            image = imageio.imread(f)
            writer.append_data(image)
        image = imageio.imread(f)
        writer.append_data(image)
