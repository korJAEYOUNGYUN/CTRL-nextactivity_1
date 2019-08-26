from model import Model
from utils import DataLoader
import torch.optim
import torch.nn
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataLoader = DataLoader(args.data_dir, args.file_name)
    x_tensor, y_tensor = dataLoader.read_data()
    x_tensor = x_tensor.clone().detach().to(device)
    y_tensor = y_tensor.clone().detach().to(device)
    dataLoader.load_preprocessed()

    model = Model(args).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    data_len = len(x_tensor)

    for i in range(args.num_epochs):
        for j in range(data_len):
            loss = 0
            optimizer.zero_grad()

            trace_len = len(x_tensor[j])
            h, c = model.init_hidden()

            for k in range(trace_len):
                h = h.to(device)
                c = c.to(device)
                x = x_tensor[j][k]
                y = y_tensor[j][k]

                if x.item() == dataLoader.word2index['END']:
                    break

                y_, (h, c) = model(x, (h, c))
                loss += loss_fn(y_.squeeze(0), y.view(1))

            loss.backward()
            optimizer.step()
            print("{}/{} (epoch {}), train_loss = {:.3f}".format(i * data_len, args.num_epochs * data_len, i* data_len + j, loss))

        if i % 10 == 9:
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.file_name + '{}.pt'.format(i+1)))


def train_from_middle(args, save_file, START, END):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataLoader = DataLoader(args.data_dir, args.file_name)
    x_tensor, y_tensor = dataLoader.read_data()
    x_tensor = x_tensor.clone().detach().to(device)
    y_tensor = y_tensor.clone().detach().to(device)
    dataLoader.load_preprocessed()

    model = Model(args)
    model.load_state_dict(torch.load(save_file + '{}.pt'.format(START)))
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    data_len = len(x_tensor)

    num_epochs = END - START

    for i in range(num_epochs):
        for j in range(data_len):
            loss = 0
            optimizer.zero_grad()

            trace_len = len(x_tensor[j])
            h, c = model.init_hidden()

            for k in range(trace_len):
                # print(k)
                h = h.to(device)
                c = c.to(device)
                x = x_tensor[j][k]
                y = y_tensor[j][k]

                if x.item() == dataLoader.word2index['END']:
                    break

                y_, (h, c) = model(x, (h, c))
                loss += loss_fn(y_.squeeze(0), y.view(1))

            loss.backward()
            optimizer.step()
            print("{}/{} (epoch {}), train_loss = {:.3f}".format(i * data_len, num_epochs * data_len,
                                                                 i * data_len + j, loss))

        if i % 10 == 9:
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.file_name + '{}.pt'.format(i+1+START)))



if __name__ == '__main__':
    import config
    args = config.Arg()
    train(args)
    #train_from_middle(args, save_file=os.path.join(args.save_dir, args.file_name), START=180, END=200)