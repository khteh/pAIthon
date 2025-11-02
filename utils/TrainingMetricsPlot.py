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
    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(20,10)) # 1 row with 2 columns; figsize = (width, height)
    fig.tight_layout(pad=5.0, rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]
    fig.suptitle(title, fontsize=22, fontweight="bold")
    axs[0].plot(x, acc, 'b', label='Training acc')
    axs[0].plot(x, val_acc, 'r', label='Validation acc')
    axs[0].set_title('Training and Validation Accuracy', fontsize=20)
    axs[0].legend(fontsize='x-large')
    axs[0].set_xlabel("Epoch", fontsize=20)
    # Set x-axis tick label size
    axs[0].tick_params(axis='x', labelsize=20) 
    # Set y-axis tick label size
    axs[0].tick_params(axis='y', labelsize=20)    

    axs[1].plot(x, loss, 'b', label='Training Loss')
    axs[1].plot(x, val_loss, 'r', label='Validation Loss')
    axs[1].set_title('Training and Validation Loss', fontsize=20)
    axs[1].legend(fontsize='x-large')
    axs[1].set_xlabel("Epoch", fontsize=20)
    # Set x-axis tick label size
    axs[1].tick_params(axis='x', labelsize=20) 
    # Set y-axis tick label size
    axs[1].tick_params(axis='y', labelsize=20)    
    plt.show()# This blocks

def PlotGANLossHistory(title, gen_losses, disc_losses):
    print(f"\n=== {PlotGANLossHistory.__name__} ===")
    fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(20, 10)) # 1 row with 1 columns; figsize = (width, height)
    fig.tight_layout(pad=5.0, rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]
    fig.suptitle(title, fontsize=22, fontweight="bold")
    x = range(1, len(gen_losses) + 1)
    axs.plot(x, gen_losses, 'b', label='Generator Loss')
    axs.plot(x, disc_losses, 'r', label='Discriminator Loss')
    axs.legend(fontsize='x-large')
    axs.set_xlabel("Epoch", fontsize=20)
    # Set x-axis tick label size
    axs.tick_params(axis='x', labelsize=20) 
    # Set y-axis tick label size
    axs.tick_params(axis='y', labelsize=20)    

    plt.show()# This blocks
