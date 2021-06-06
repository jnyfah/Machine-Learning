import random 
import numpy as np 
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape 
from numpy import exp, gradient


class neuralNetworks:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes                          #convert input to single column arrary
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = 0.1

        self.weights_IH = np.random.rand (self.hidden_nodes, self.input_nodes)   #initialize random weights
        self.weights_HO = np.random.rand(self.output_nodes, self.hidden_nodes)

        self.bias_H = np.random.rand(self.hidden_nodes, 1)
        self.bias_O = np.random.rand(self.output_nodes, 1)


    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def dsigmoid(self, x):
        #return (self.sigmoid(x) * (1 - self.sigmoid(x)))    #derivative of sigmoid function
        return (x * (1-x))

    
    def feedFoward(self, input):
        input = np.reshape(input,(len(input), 1))   #convert input to matrix

        hidden = np.dot(self.weights_IH, input)
        hidden = np.add(hidden, self.bias_H)
        hidden = self.sigmoid(hidden)

        output = np.dot(self.weights_HO, hidden)
        output = np.add(output, self.bias_O)
        output = self.sigmoid(output)

        return output

    def train(self, input, target):                                 #backpropagation
        input = np.reshape(input,(len(input), 1))   #convert input to matrix
        target = np.reshape(target,(len(target), 1))   #convert input to matrix
        
        #Feedfoward layer 1
        hidden = np.dot(self.weights_IH, input)
        hidden = np.add(hidden, self.bias_H)
        hidden = self.sigmoid(hidden)
         
        #layer 2
        output = np.dot(self.weights_HO, hidden)
        output = np.add(output, self.bias_O)
        output = self.sigmoid(output)

        #output errors
        error = np.subtract(target, output)

        #backpropagation layer 2
        gradient = np.dot(self.learning_rate, error,  self.dsigmoid(output))
        #Adjust the bias
        self.bias_O = np.add(self.bias_O, gradient)                              #Bias += learning rate * Error * GD
        weights_HO_deltas = np.dot(gradient, hidden.transpose())
        self.weights_HO = np.add(self.weights_HO, weights_HO_deltas)

        #calculate hidden layer errors
        weights_THO = self.weights_HO.transpose()
        hidden_error = np.dot(weights_THO, error)

        #backpropagation layer 1
        hidden_gradients = np.dot(self.learning_rate, hidden_error, self.dsigmoid(hidden))
        #Adjust the bias
        self.bias_H = np.add(self.bias_H, hidden_gradients)
        weights_IH_deltas = np.dot(hidden_gradients, input.transpose())
        self.weights_IH = np.add(self.weights_IH, weights_IH_deltas)
    
        
        





def mainf():
    nn = neuralNetworks(2,2,1)
    training_data = np.array([[1,0],
                              [0,1], 
                              [1,1], 
                              [0,0]])

    target_data = np.array([[1],
                       [1],
                       [0],
                       [0]])
    
    for i in range (50000):
        
            #nn.train([1,1], [0])
            nn.train([1,0], [1])
            nn.train([0,1], [1])
            nn.train([0,0], [0])
            

   
    print(nn.feedFoward([1,0]))
    print(nn.feedFoward([0,1]))
    print(nn.feedFoward([0,0]))
    print(nn.feedFoward([1,1]))

def main():
    target = [1,2,3]
    target = np.reshape(target,(len(target), 1))
    print(target)

mainf()