import numpy as np

class maxPool2:

    def iterateRegions(self, image):

        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):

                imRegion = image[(i*2):(i*2+2), (j*2):(j*2+2)]
                yield imRegion, i ,j                                       #works the same as teh convo3x3 class, this is the generator

    def forward(self, input):

        self.lastInput = input

        h, w, numFilters = input.shape
        output = np.zeros((h // 2, w // 2, numFilters))                     #once again is a vector of zeros with shape of dimensions specificed in (())

        for imRegion, i, j in self.iterateRegions(input):
            output[i,j] = np.amax(imRegion, axis=(0,1))                     #amax gets the largest valye within the specifeid array.

        return output

    def backProp(self, dLdout):

        dLdinp = np.zeros(self.lastInput.shape)

        for imRegion, i, j  in self.iterateRegions(self.lastInput):
            h, w, f = imRegion.shape
            amax = np.amax(imRegion, axis=(0, 1))                           # get the amax [largest value] of each "page"

            for i2  in range(h):

                for j2 in range(w):

                    for f2 in range(f):

                        if imRegion[i2,j2,f2] == amax[f2]:
                            dLdinp[i * 2 + i2, j * 2 + j2, f2] = dLdout[i,j,f2]
        
        return dLdinp
                            