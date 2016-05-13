##########Geo Regression####################################################################################
# Geo regression is a regression method for regression problems based on the geographical locations, but   #
# it is not limited to such application. Essentially, any application that wants to have smoothed parameter#
# transitions between "neighbors" [defined by a ajacency matrix] can use such method.                      #
############################################################################################################




import numpy as np
import theano



def train():
    nRegions = 10
    adjacencyMatrix = np.zeros((nRegions, nRegions))
