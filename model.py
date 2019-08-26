import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module) :
    def __init__(self, args):
        super(Model, self).__init__()

        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.output_size = args.output_size
        self.num_layers = args.num_layers
        self.embedding_size = args.embedding_size
        self.batch_size = args.batch_size

        self.embedding = nn.Embedding(args.embedding_size, args.embedding_size)
        self.lstm = nn.LSTM(input_size=args.embedding_size, hidden_size=args.hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(args.hidden_size, args.output_size)
        #self.softmax = nn.Softmax()


    def forward(self, input, hc):

        embedded = self.embedding(input)
        output, hc = self.lstm(embedded.view(self.batch_size, self.input_size, -1), hc)
        output = self.fc(output)
        #output = self.softmax(output)

        return output, hc


    def init_hidden(self):
        h = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        c = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

        return h, c
