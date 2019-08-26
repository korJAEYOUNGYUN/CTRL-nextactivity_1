import os
import torch
from six.moves import cPickle
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

class DataLoader():
    def __init__(self, data_dir, file_name):
        self.word2index = {}     #dict{word , index}
        self.index2word = {}     #dict{index , word}
       # self.indexed_input_data = []     #입력 데이터인 워크케이스의 액티비티 ID를 인덱스로 교체한 데이터

        self.n_words = 0
        self.data_dir = data_dir

        self.input_file = os.path.join(data_dir, file_name)
        self.word2index_file = os.path.join(data_dir, file_name + "word2index.pkl")        #인덱스 파일 word2index 저장
        self.index2word_file = os.path.join(data_dir, file_name + "index2word.pkl")        # index2word 저장
        self.tensor_file = os.path.join(data_dir, file_name + ".npy")       #텐서 파일 tensor 저장

        self.x_data = []
        self.y_data = []


    def read_data(self):
        with open(self.input_file, "r") as f:
            data = f.read()

        data = data[:-1]
        sentences = data.split("\n")

        for sentence in sentences:
            self.index_words(sentence)

        self.input_tensor = pad_sequence(self.x_data, batch_first=True, padding_value=self.word2index['END'])
        self.target_tensor = pad_sequence(self.y_data, batch_first=True, padding_value=self.word2index['END'])

        with open(self.word2index_file, 'wb') as f:
            cPickle.dump(self.word2index, f)

        with open(self.index2word_file, 'wb') as f:
            cPickle.dump(self.index2word, f)

        self.word2index.clear()

        #self.input_tensor = self.one_hot(self.input_tensor)

        return self.input_tensor, self.target_tensor


    def index_words(self, sentence):
        self.indexed_input_data = []
        self.indexed_input_data.clear()

        for word in sentence.split('->'):
            self.index_word(word)

        self.x_data.append(torch.LongTensor(self.indexed_input_data[:-1]))
        self.y_data.append(torch.LongTensor(self.indexed_input_data[1:]))


    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

        self.indexed_input_data.append(self.word2index[word])


    def load_preprocessed(self):
        with open(self.word2index_file, 'rb') as f:
            self.word2index = cPickle.load(f)

        with open(self.index2word_file, 'rb') as f:
            self.index2word = cPickle.load(f)

        return self.word2index, self.index2word


    def one_hot(self, tensor, num_classes=-1):
        return F.one_hot(tensor, num_classes=num_classes)


# dataLoader = DataLoader("data", "review_example_large")
# word2index, _ = dataLoader.load_preprocessed()
# print(dataLoader.word2index['END'])
# x_tensor, y_tensor = dataLoader.read_data()
# # print(x_tensor.shape)
# # print(y_tensor.shape)
# # print(x_tensor[0].shape)
# # print(y_tensor[0])
#
# import torch.nn.functional as F
#
# print(F.one_hot(x_tensor).size())
# print(F.one_hot(x_tensor)[0].size())
# print(F.one_hot(x_tensor)[0][0])
#
# print(torch.eye(16)[x_tensor, :].size())

# import torch.nn
# emb = torch.nn.Embedding(16,16)
# x = emb(x_tensor[0][10])
# print(x)
