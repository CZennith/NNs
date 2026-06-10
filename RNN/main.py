import numpy as np
import random

from rnn import RNN
from data import test_data, train_data

vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocabSize = len(vocab)

wordToIndex = { w: i for i, w in enumerate(vocab)}
indexToWord = { i: w for i, w in enumerate(vocab)}

def createInputs(text):

    inputs = []
    for w in text.split(' '):
        v = np.zeros((vocabSize, 1))                    #initialises a vector of 0s of columns equal to the number of words and one row.
        v[wordToIndex[w]] = 1                           #using the dict from earlier to retrive the word's index and replace the vector's corresponding index with 1
        inputs.append(v)
    return inputs


def softMax(xs):

    return np.exp(xs) / sum(np.exp(xs))

rnn = RNN(vocabSize, 2)                                 #inputsize of vocab size, output size of 2

# inputs = createInputs("i am very good")
# out, h = rnn.forward(inputs)

# probs = softMax(out)                                    #returns the probability distribution between the output and hidden node

# for x, y in train_data.items():                         #each item is the key + content

#     inputs = createInputs(x)
#     target = int(y)

#     #Forward prop
#     out, _ = rnn.forward(inputs)                        # _ because we do not need the vector, only the result
#     probs = softMax(out)

#     #dL/dy
#     dLdy = probs                                        #using cross entropy, the derivate of the loss = prob of output at the correct class
#     dLdy[target] -= 1                                   #else it is -1 of the probability

#     rnn.backProp(dLdy)


def processData(data, backProp=True):

    items = list(data.items())
    random.shuffle(items)

    loss = 0
    numCorrect = 0

    for x, y in items:                                  #again, x is the key, y is the bool
        inputs = createInputs(x)
        target = int(y)
        
        #Forward feed
        out, _ = rnn.forward(inputs)
        probs = softMax(out)

        #Find loss 
        loss -= np.log(probs[target])                   #gradient = -log(p)
        numCorrect += int(np.argmax(probs) == target)   #argmax converts prov to 1 or 0

        if backProp:

            dLdy = probs
            dLdy[target] -= 1

            #backgeed
            rnn.backProp(dLdy)
    
    return loss / len(data), numCorrect / len(data)


for epoch in range(3000):
    
    train_loss, train_acc = processData(train_data)

    if epoch % 100 == 99:
        print("--- Epoch %d" % (epoch + 1))
        print("Train:\tLoss %.3f | Accuracy: %.3f" % (train_loss.item(), train_acc))

        test_loss, test_acc = processData(test_data, backProp = False)
        print("Test:\tLoss %.3f | Accuracy: %.3f" % (test_loss.item(), test_acc))
