from multiLayerPerceptron import neuralNetworks
import numpy as np

def main():
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
    


main()

