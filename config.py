import os

class Arg():
    def __init__(self):
        #for load and save
        self.data_dir = 'data'
        self.file_name = 'review_example_large'
        self.save_dir = os.path.join('save', 'review example')
        self.log_dir = os.path.join('logs', 'review example')


        #for neural network
        self.hidden_size = 128
        self.num_layers = 2
        self.input_size = 1
        self.output_size = 16
        self.embedding_size = 16


        #training hyperparameter
        self.num_epochs = 50
        self.learning_rate = 0.002
        self.batch_size = 1


