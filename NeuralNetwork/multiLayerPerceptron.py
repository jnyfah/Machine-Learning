import random 
import math
import numpy as np 
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape 
from numpy import exp


class neuralNetworks:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes                          #convert input to single column arrary
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_IH = np.random.rand (self.hidden_nodes, self.input_nodes)   #initialize random weights
        self.weights_HO = np.random.rand(self.output_nodes, self.hidden_nodes)

        self.bias_H = np.random.rand(self.hidden_nodes, 1)
        self.bias_O = np.random.rand(self.output_nodes, 1)


    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    
    def feedFoward(self, input):
        input = np.reshape(input,(len(input), 1))

        hidden = np.dot(self.weights_IH, input)
        hidden = np.add(hidden, self.bias_H)
        hidden = self.sigmoid(hidden)

        output = np.dot(self.weights_HO, hidden)
        output = np.add(output, self.bias_O)
        output = self.sigmoid(output)

        return output



def main():
    nn = neuralNetworks(2,2,1)
    input = [1,0]
    output = nn.feedFoward(input)
    print (output)

main()