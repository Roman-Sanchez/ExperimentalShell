################################################################
# The DecisionTree class is a classifier inheriting from the base
# class Classifier.
#
################################################################

# Imports
from Classifier import Classifier
import numpy as np
import pandas as pd
from sklearn import datasets


# End of Imports

class DecisionTree(Classifier):
    # Init function
    def __init__(self):
        self.name = "Decision Tree ID3"
        self.tree = None

    # Calculates the entropy
    def entropy(self, num):
        if num != 0:
            return -num * np.Log2(num)
        else:
            return 0

    # This Function will make the tree recursively
    def makeTree(self, data, classes, featureNames):
        numData = len(data)
        numFeatures = len(featureNames)
        default = classes[0]

        if numData == 0 or numFeatures == 0:
            return default
        # if the classes count equals that of numData then there is only one class left so return that class
        elif classes.count(classes[0]) == numData:
            return classes[0]
        # choose which feature is best
        else:
            pass

    # Calculates the info gain
    def calcInfoGain(self):
        pass

    def outputTree(self):
        pass
