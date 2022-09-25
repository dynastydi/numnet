import numpy as np


def sigmoid(data):

    # activation function for the net.
    
    return 1 / (1 + np.exp(- data))

# the network as an object.

class net:
    def __init__(self, input : int, hidden : int, output : int, learning_rate):
        self.learning_rate = learning_rate

        # weights mapping between layers ( from, to ):
        # array dimensions are FROM by TO
        # initialised values are of uniform distributions between:
        #           -0.05 and 0.05
        self.h_weights = (np.random.random((input, hidden)) - 0.5).astype(float) / 10
        self.o_weights = (np.random.random((hidden, output)) - 0.5).astype(float) / 10

    def predict(self, data):
        h_out = sigmoid(data.dot(self.h_weights))
        o_out = sigmoid(h_out.dot(self.o_weights))
        return h_out, o_out

    def train(self, in_data, targ_data):

        # forward propagate
        
        h_out, o_out = self.predict(in_data)


        o_errors = o_out - targ_data    # output errors
        h_errors = o_errors.dot(self.o_weights.T) * h_out

        o_gradient = h_out.T.dot(o_errors)
        h_gradient = in_data.T.dot(h_errors)

        self.o_weights -= self.learning_rate * o_gradient
        self.h_weights -= self.learning_rate * h_gradient

    def save(self, h_path, o_path):
        np.save(h_path, self.h_weights)
        np.save(o_path, self.o_weights)

    def load(self, h_path, o_path):
        self.h_weights = np.load(h_path)
        self.o_weights = np.load(o_path)
