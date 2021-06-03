import math
import random

class Perceptron:
    def __init__(self, n):
        self.weights = []
        self.sum = 0
        self.output = 0
        self.lr = 0.1

        for i in range(n):
            self.weights.append(round(random.uniform(-1.00, 1.00)))

    def sign(self, n):
        if(n >= 0.00):
            return 1
        else:
            return -1

    def weightedSum(self, inputs):
        for i in range(len(self.weights)):
            self.sum += self.weights[i] * inputs[i]
        self.output = self.sign(self.sum)
        return self.output

    def train(self, inputs, target):
        pred = self.weightedSum(inputs)
        error = target - pred

        for i in range (len(self.weights)):
            self.weights[i] = error * inputs[i] * self.lr
    


def main():
    inputs =[-1, 6]
    Pep = Perceptron(2)
    x = Pep.weightedSum(inputs)
    print(x)

main()







