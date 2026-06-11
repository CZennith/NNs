import numpy as np

class Conv3x3:

    def __init__(self, numFilters):

        self.numFilters = numFilters

        self.filters = np.random.randn(numFilters, 3, 3) / 9            #this filter is a 3d array with numF rows, 3 height and 3 width
                                                                        #it is divided by 9 to reduce the variance between the initial values
                                                                        #this is important to make sure that the NN can be properly trained. 
    
    #image is a 2d numpy array
    def iterateRegions(self, image):
        h, w = image.shape

        #since the iamge is iterated throuhg with valid padding, the center of the filter will start and end at on the 2nd "element" from each side
        for i in range(h - 2):

            for j in range(w - 2):

                imRegion = image[i:(i+3), j:(j+3)]                      #taking a 3x3 cut out with top left row i, col j
                yield imRegion, i, j                                    #yield and return both return a value, but return termiantes the function while yield pauses it. It's a generator
                                                                        #when it is called again, yield starts off where it previously returned while return will start from the very beginning
    

    #forward feeding
    def forward(self, input):
        #takes in 2d numpy array and returns a 3d numpy array of h x w x num filters

        h, w = input.shape
        output = np.zeros((h-2, w-2, self.numFilters))                  #inititiate vector filled with zeros of shape h-2 x w-2 x number of filters

        for imRegion, i, j in self.iterateRegions(input):

            #imRegion is the generated image array earlier, multiplied with the filter's values 
            #to create the convoluted image
            output[i,j] = np.sum(imRegion * self.filters, axis=(1,2))   #axis(1,2) means that the funciton will take the sum of all elements in each depth (sum of all pixel value in each convoluted image), 
                                                                        #resulting in 1d array with length num filters

            #output[i,j] contains the convolution result for pixel i,j

        return output

