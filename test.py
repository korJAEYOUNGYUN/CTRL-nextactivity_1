from model import Model
from utils import DataLoader
import torch
import os
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def test(args):
    save_file = os.path.join(args.save_dir, args.file_name + '200.pt')
    model = Model(args)
    model.load_state_dict(torch.load(save_file))
    model.eval()

    dataLoader = DataLoader(args.data_dir, args.file_name)
    word2index, index2word = dataLoader.load_preprocessed()

    history = []
    hc = model.init_hidden()

    while True:

        activity = input(': ')
        history.append(activity)
        print('input history:')
        for i in range(len(history)):
            print(history[i], end=' ')
        print()

        x = word2index[activity]
        input_tensor = torch.tensor(x)

        y_, hc = model(input_tensor, hc)
        print('predicted activity: ')
        print(index2word[torch.argmax(F.softmax(y_, dim=2)).item()])
        i = 0
        probabilities = []
        for x in F.softmax(y_, dim=2)[0][0]:
            probabilities.append(x.item() * 100)

        y_pos = np.arange(len(index2word))
        plt.barh(y_pos, probabilities)
        plt.yticks(y_pos, index2word.values())
        plt.title('next activities')
        plt.xlabel('probabilities')

        plt.subplots_adjust(left=0.3)

        plt.show()

if __name__ == '__main__':
    import config
    args = config.Arg()
    test(args)