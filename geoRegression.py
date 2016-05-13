##########Geo Regression####################################################################################
# Geo regression is a regression method for regression problems based on the geographical locations, but   #
# it is not limited to such application. Essentially, any application that wants to have smoothed parameter#
# transitions between "neighbors" [defined by a ajacency matrix] can use such method.                      #
############################################################################################################




import numpy as np
import theano
import theano.tensor as T
import theano.typed_list
import pandas as pd


def train():
    nRegions = 2
    adjacencyMatrix = np.zeros((nRegions, nRegions))

    # The shape of the inputData is [nRregions, nx(p + 1)]
    inputData = theano.typed_list.TypedListType(T.fmatrix)()

    # The shape of the inputDataResponse is [nRegions, n]
    inputDataResponse = theano.typed_list.TypedListType(T.fvector)()


    coefficient = T.fmatrix()
    loop, _ = theano.scan(fn = lambda i, tl:tl[i].sum(),non_sequences=[inputData], sequences=[theano.tensor.arange(nRegions, dtype = 'int64')])

    #predictions = [T.dot(inputData[i], coefficient[i]) for i in range(nRegions)]
    #rss = np.sum([(T.sqr(predictions[i] - inputDataResponse[i])).sum() for i in range(nRegions)])

    output_function = theano.function([inputData], [loop])



    loaded_data = [np.zeros((3, 5), dtype = np.float32) for i in range(2)]
    #loded_data_label = [np.zeros(3) for i in range(2)]
    #loaded_data_coeffcient = [np.zeros(5)]
    print output_function(loaded_data)






def main():
    train()

if __name__ == "__main__":
    main()




