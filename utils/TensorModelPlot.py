import matplotlib.pyplot as plt
plt.style.use('ggplot')

def PlotModelHistory(title, history):
    print(f"\n=== {PlotModelHistory.__name__} ===")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(title)
    fig.set_size_inches(12, 5) # (w, h) inches
    fig.set_layout_engine('constrained')
    axs[0].plot(x, acc, 'b', label='Training acc')
    axs[0].plot(x, val_acc, 'r', label='Validation acc')
    axs[0].set_title('Training and validation accuracy')
    axs[0].legend()
    axs[1].plot(x, loss, 'b', label='Training loss')
    axs[1].plot(x, val_loss, 'r', label='Validation loss')
    axs[1].set_title('Training and validation loss')
    axs[1].legend()
    plt.show()# This blocks