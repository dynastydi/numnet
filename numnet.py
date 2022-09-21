import numpy as np


def sigmoid(data):

    # activation function for the net.
    
    return 1 / (1 + np.exp(- data))

def prime(data):
    return data * (1 - data)

# the network as an object.

class net:
    def __init__(self, input : int, hidden : int, output : int, learning_rate):
        self.input = input
        self.hidden = hidden
        self.output = output
        self.learning_rate = learning_rate

        # weights mapping between layers ( from, to ):
        # array dimensions are FROM by TO
        # initialised values are of uniform distributions between:
        #           -0.05 and 0.05
        self.h_weights = (np.random.random((hidden, input)) - 0.5).astype(np.single) / 10
        self.o_weights = (np.random.random((output, hidden)) - 0.5).astype(np.single) / 10

    def predict(self, data):
        h_out = sigmoid(self.h_weights.dot(data))
        o_out = sigmoid(self.o_weights.dot(h_out))
        return h_out, o_out

    def train(self, in_data, targ_data):

        # forward propagate
        
        h_out, o_out = self.predict(in_data)

        # output errors
        o_errors = targ_data - o_out
        # hidden errors through the dot product
        h_errors = np.transpose(self.o_weights).dot(o_errors)
        
        # backpropagate
        self.o_weights += self.learning_rate * (o_errors * prime(o_out)).dot(np.transpose(h_out))
        self.h_weights += self.learning_rate * (h_errors * prime(h_out)).dot(np.transpose(in_data))

    def save(self, h_path, o_path):
        np.save(h_path, self.h_weights)
        np.save(o_path, self.o_weights)

    def load(self, h_path, o_path):
        self.h_weights = np.load(h_path)
        self.o_weights = np.load(o_path)
