import mnist                                                  #used to gather the training data for the project
import numpy as np
from conv import Conv3x3
from maxpool import maxPool2
from softmax import softMax

testImages = mnist.test_images()[:1000]
testLabels = mnist.test_labels()[:1000]


#create conv calss for 8 filters
conv = Conv3x3(8)                                               #28x28x1 > 26x16x8
maxp = maxPool2()                                               #26x26x8 > 13x13x8          #pools together nearby pixels in a 2x2 to get the largest pixel value. Since in the convoluted image, nearby pixels have similar values
softmax = softMax(13*13*8, 10)                                  #13x13x8 > 10 (nodes)

def forward(image, label):

    #forward feeding, calcs loss (cross entropy same as in RNN) and acc

    out = conv.forward((image / 255) - 0.5)                    #converts pixel values from 0 tp 255 to -0.5 to 0.5. Standard practice for training
    out = maxp.forward(out)
    out = softmax.forward(out)


    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

print("MNIST CNN initialised")

loss = 0
numCorrect = 0

for i, (im, label) in enumerate(zip(testImages, testLabels)):

    _, l, acc = forward(im,label)
    loss += 1
    numCorrect += acc

    if i % 100 == 99:
        print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %(i + 1, loss / 100, numCorrect))
    
        loss = 0
        numCorrect = 0                          



