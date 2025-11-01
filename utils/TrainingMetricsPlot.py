import matplotlib.pyplot as plt
plt.style.use('ggplot')

def PlotModelHistory(title, history):
    print(f"\n=== {PlotModelHistory.__name__} ===")
    if "accuracy" not in history.history:
        print("No accuracy metric in history!")
        return
    if "val_accuracy" not in history.history:
        print("No val_accuracy metric in history!")
        return   
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(10,10)) # 1 row with 2 columns; figsize = (width, height)
    fig.suptitle(title)
    fig.set_size_inches(12, 5) # (w, h) inches
    fig.set_layout_engine('constrained')
    axs[0].plot(x, acc, 'b', label='Training acc')
    axs[0].plot(x, val_acc, 'r', label='Validation acc')
    axs[0].set_title('Training and validation accuracy')
    axs[0].legend()
    axs[0].set_xlabel("Epoch")
    axs[1].plot(x, loss, 'b', label='Training loss')
    axs[1].plot(x, val_loss, 'r', label='Validation loss')
    axs[1].set_title('Training and validation loss')
    axs[1].legend()
    axs[1].set_xlabel("Epoch")
    plt.show()# This blocks

def PlotGANLossHistory(title, gen_losses, disc_losses):
    print(f"\n=== {PlotGANLossHistory.__name__} ===")
    fig, axs = plt.subplots(1, 1, figsize=(10,10)) # 1 row with 1 columns; figsize = (width, height)
    fig.suptitle(title)
    fig.set_size_inches(12, 5) # (w, h) inches
    fig.set_layout_engine('constrained')
    x = range(1, len(gen_losses) + 1)
    axs.plot(x, gen_losses, 'b', label='Generator loss')
    axs.plot(x, disc_losses, 'r', label='Discriminator loss')
    axs.legend()
    axs.set_xlabel("Epoch")
    plt.show()# This blocks
