import numpy as np
from numpy.random import randn

class RNN:
    #Basic recurrent Neural Network

    def __init__ (self, inputSize, outputSize, hiddenSize=64):

        self.Whh = randn(hiddenSize, hiddenSize) / 1000      #Weight between hidden Nodes (h -> h)
        self.Wxh = randn(hiddenSize, inputSize) / 1000       #Weight between input to hidden Nodes (x -> h)
        self.Why = randn(outputSize, hiddenSize) / 1000      #Weight between hidden Nodes and output (h -> y)

        self.bh = np.zeros((hiddenSize, 1))                         #vectors of zeros
        self.by = np.zeros((outputSize, 1))

    def forward(self, inputs):

        #Returns the final output and hidden state

        h = np.zeros((self.Whh.shape[0], 1))

        self.lastInputs = inputs
        self.lastHs = {0: h}

        #i is the index of the enumerated, x is the word
        for i, x in enumerate(inputs):

            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)      # @ is the matrix multiplication operator
            self.lastHs[i + 1] = h

        y = self.Why @ h + self.by

        return y, h

    def backProp(self, dY, learnRate=2e-3):

        #dY is dL/dy with shape (outputSize, 1)

        n = len(self.lastInputs)

        #y = weight * hidden + bias
        #so since dL/dWhy = dL / dy * dy / dWhy, dL / dWhy = dL / dy * hidden
        #same for dL/dBy, resulting in dL / dy 


        dWhy = dY @ self.lastHs[n].T           #get latest input,  .T transposes the vector 
        dby = dY
        
        #Initilise remaing gradients to zero

        dWhh = np.zeros(self.Whh.shape)
        dWxh = np.zeros(self.Wxh.shape)
        dbh = np.zeros(self.bh.shape)

        #final hidden node gradient
        dh = self.Why.T @ dY

        for t in reversed(range(n)):
            
            #since h is a tanh(Wxh * Xt + Whh * ht-1 + bh)
            #dh/dx = 1 - tanh^2(x), dh/dWxh = dh / dx * dx / dWxh = (1 - tanh^2(x))Xt
            #again, dh / dWhh = (1 - tanh^2(x))(ht-1), and dh / dbh = 1 - tanh^2(x) 
            #excetpf or the last hidden node, where dy/dh = Why, 
            #dy / dht = dy /dht+1 * dht+1 / dh =  (1 - tanh^2(x))Whh 

            #temp dL/dh * (1 - h^2)
            temp = ((1 - self.lastHs[t + 1] ** 2) * dh)

            # dL/dbh =  dL/dh * (1 - h^2)
            dbh += temp

            # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
            dWhh += temp @ self.lastHs[t].T

            #dL/dWxh = dL / dh * (1 - h^2) * x
            dWxh += temp @ self.lastInputs[t].T

            #dL/dh = dL/dh * (1 - h^2) * Whh
            dh = self.Whh @ temp

            #to prevent exploding gradeints by limitting the max to 1 and min to -1
            for d in [dWxh, dWhh, dWhy, dby, dbh]:
                np.clip(d, -1, 1, out=d)
            
            #update wights and biasse
            self.Whh -= learnRate * dWhh
            self.Why -= learnRate * dWhy
            self.Wxh -= learnRate * dWxh
            self.bh -= learnRate * dbh
            self.by -= learnRate * dby
