################################################################
# The KNN class is a classifier that will implement the K-Nearest
# Neighbor algorithm.
#
################################################################
import numpy as np
from collections import Counter


class KNN(object):
    predictionResults = []
    distances = []
    fittedData = []
    testAmount = 0

    def __init__(self):
        self.name = "KNN Classifier"

    # Trains the program
    def fit(self, trainingData, targetTrain):

        for item, target in zip(trainingData, targetTrain):
            pair = item, target
            self.fittedData.append(pair)

        return "Done been trained"

    # Takes in a set of data and makes a prediction
    # Returns an array of the predictions
    def predict(self, k, dataTest):
        if k <= len(self.fittedData):
            self.testAmount = dataTest.size

            for row in dataTest:

                # calculate distances
                self.calcDistances(row)

                # returns all of the indices from the distances array.
                # indices is sorted by min
                indices = np.argsort(self.distances)
                classes = []

                for n in range(k):
                    # using the index from indices get the target value from the fittedData
                    # fittedData is a tuple with the second element being the target class.
                    classes.append(self.fittedData[indices[n]][1])

                self.predictionResults.append(self.predictClass(classes))

                # after a prediction has been made, set the member distance list to empty
                # so that the previous stored distances will be cleared out.
                self.distances.clear()

        return self.predictionResults

    # Handles all of the prediction logic when the classes have been identified
    def predictClass(self, classes):
        mostCommon = Counter(classes).most_common(1)

        # mostCommon is a tuple that is set
        prediction = mostCommon[0][0]

        return prediction

    # Calculates all of the distances between a row from the test data and all of the rows
    # in the fitted data
    def calcDistances(self, row):
        distance = 0

        # iterate over all of the fitted data and calculate all distances
        for fittedRow in self.fittedData:

            if fittedRow[0].size == row.size:
                for x, y in zip(fittedRow[0], row):
                    distance += (x - y)**2

                # append the distance to the distance array
                self.distances.append(distance)

                # Reinitialize distance to 0
                distance = 0



    #
    #
    def calcAccuracy(self, targetTest):
        accuracy = 0.0
        if len(targetTest) == len(self.predictionResults):
            numCorrect = 0
            numIncorrect = 0
            counter = 0
            for item in targetTest:
                if item == self.predictionResults[counter]:
                    numCorrect += 1
                else:
                    numIncorrect += 1

                counter += 1
            accuracy = (numCorrect / len(self.predictionResults)) * 100

        return "%.2f" % accuracy