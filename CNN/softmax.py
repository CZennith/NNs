import numpy as np

class softMax:

    def __init__(self, inputLen, nodes):

        self.weights = np.random.randn(inputLen, nodes) / inputLen              #divided to reduce the variance between each valye
        self.bias = np.zeros(nodes)                                             #biases in NNs are matrixes

    
    def forward(self, input):


        input = input.flatten()                                             #flattened since image no longer needed [converts whatever array into 1d array of length]

        inputLen, nodes = self.weights.shape

        #activation function
        totals = np.dot(input, self.weights) + self.bias                    #dot product of matrixes, element wise multiplication
        exp = np.exp(totals)                                                #get softmax

        return exp / np.sum(exp, axis = 0)                                  #returns 1d numpy array of  respective probability values
