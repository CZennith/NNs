import numpy as np

class softMax:

    def __init__(self, inputLen, nodes):

        self.weights = np.random.randn(inputLen, nodes) / inputLen              #divided to reduce the variance between each valye
        self.bias = np.zeros(nodes)                                             #biases in NNs are matrixes

    
    def forward(self, input):

        self.lastInputShape = input.shape                                   #cache shape

        input = input.flatten()                                             #flattened since image no longer needed [converts whatever array into 1d array of length]
        self.lastInput = input                                              #cache flattened shape

        inputLen, nodes = self.weights.shape

        #activation function
        totals = np.dot(input, self.weights) + self.bias                    #dot product of matrixes, element wise multiplication
        self.lastTotal = totals                                             #predicted value of each image cached
        
        exp = np.exp(totals)                                                #get softmax

        return exp / np.sum(exp, axis = 0)                                  #returns 1d numpy array of  respective probability values

    def backProp(self, dLdout, learnRate):


        #only the correct answer's gradient will be non-zero, therefore:
        for i, gradient in enumerate(dLdout):
            if gradient == 0: continue

            totalExp = np.exp(self.lastTotal)

            Se = np.sum(totalExp)

            # print("lastTotal =", self.lastTotal)
            # print("max total =", np.max(self.lastTotal))
            # print("min total =", np.min(self.lastTotal))

            # totalExp = np.exp(self.lastTotal)

            # print("Any inf?", np.isinf(totalExp).any())
            # print("Se =", np.sum(totalExp))

            dOutdt = -totalExp[i] * totalExp / ( Se ** 2)
            dOutdt[i] = totalExp[i] * (Se - totalExp[i]) / ( Se ** 2)       #accidentally placed a minus sign causing the model to go haywire. Had to use chat to debug

            #gradient of sum above against weights, biases and inputs 
            dtdw = self.lastInput
            dtdb = 1
            dtdInp = self.weights

            dLdt = gradient * dOutdt                                        #Gradient of the loss (actual - prediction) against totals 

            #gradient of the loss against the 3 parameters above
            dLdw = dtdw[np.newaxis].T @ dLdt[np.newaxis]                    #dtdw is only 1d array (was flattened) > uses newaxis to convert it into array of (inputLen, 1) and (1, node)
            dLdb = dLdt * dtdb
            dLdInp = dtdInp @ dLdt                                          #multiply matrix of (inputLen, nodes)[dtdInp] and (nodes, 1)[dLdt]


            #update weights and biases
            self.weights -= learnRate * dLdw
            self.bias -= learnRate * dLdb
            return dLdInp.reshape(self.lastInputShape)                      #reshape the flattened input to ensure that this layer returns a gradient in the same format as the input gradient
