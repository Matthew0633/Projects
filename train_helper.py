import os
import matplotlib.pyplot as plt

def viz_history(history, save = False):
    # acc, loss visualization (for 'epoch' optimization)
    acc = history.history['acc']
    loss = history.history['loss']
    x = range(len(acc))

    plt.plot(x, acc, 'b', label='accuracy')
    plt.plot(x, loss, 'r', label='loss')
    plt.title('accuracy and loss')
    plt.legend()
    plt.show()

    if save:
        if not os.path.isdir('./history'):
            os.mkdir('./history')

        plt.savefig('history/train_history.png')