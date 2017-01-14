# Inherits from the Classifier class
from Classifier import Classifier


class HardcodedClassifier(Classifier):

    predictionResults = []
    testAmount = 0

    def __init__(self):
        self.name = "Hardcoded Classifier"

    # Trains the program
    def fit(self, trainingData, targetTrain):
        return "Done been trained"

    # Takes in a set of data and makes a prediction
    # Returns an array of the predictions
    def predict(self, dataTest):
        self.testAmount = dataTest.size

        for item in dataTest:
            prediction = 0
            self.predictionResults.append(prediction)

        return self.predictionResults

    #
    #
    def calcAccuracy(self, targetTest):
        numCorrect = 0
        numIncorrect = 0
        counter = 0
        for item in targetTest:
            if item == self.predictionResults[counter]:
                numCorrect += 1
            else:
                numIncorrect += 1

            counter += 1

        return (numCorrect / len(self.predictionResults)) * 100
