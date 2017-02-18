# Runs the main program
from HardcodedClassifier import HardcodedClassifier
from DecisionTree import DecisionTree
from KNN import KNN
from NeuralNetwork import NeuralNetwork
from sklearn import datasets
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def prompt():

    print("Which classifier do you wish to run?")
    print("KNN: Enter knn")
    print("ID3 coming soon")
    print("Neural Network: Enter nn")

    answer = input("Enter: ")

    if answer == "knn":
        startKNN()
    elif answer == "nn":
        startNNetwork()
    else:
        print("Game over!")


def startNNetwork():

    NNetworkClassifier = NeuralNetwork([2, 4, 2], 4, 0.1, 3)
    NNetworkClassifier.build_layers()
    print("Built layers")

def startKNN():

    iris = datasets.load_iris()

    KNNClassifier = KNN()
    dataTrain, dataTest, targetTrain, targetTest = train_test_split(iris.data, iris.target, test_size=0.3)

    # Normalize data
    std_scale = preprocessing.StandardScaler().fit(dataTrain)
    dataTrain = std_scale.transform(dataTrain)
    dataTest = std_scale.transform(dataTest)

    # Prompt the user for number of neighbors
    kString = input("Enter K: ")

    try:
        k = int(kString)

        # Train the program
        KNNClassifier.fit(dataTrain, targetTrain)

        # Make predictions using test data
        KNNClassifier.predict(k, dataTest)

        # Print out the accuracy
        print(KNNClassifier.calcAccuracy(targetTest), "% Correct")
    except ValueError:
        print("Must enter a number")


# starter method
def main():
    # myClassifier = HardcodedClassifier()

    # iris = datasets.load_iris()

    prompt()
    # Show the data (the attributes of each instance)
    # print(iris.data)
    #
    # # Show the target values (in numeric format) of each instance
    # print(iris.target)
    #
    # # Show the actual target names that correspond to each number
    # print(iris.target_names)
    #

    # dataTrain, dataTest, targetTrain, targetTest = train_test_split(iris.data, iris.target, test_size=0.3)

    ####################################################################
    ####################################################################
    ####################################################################
    ####################### Hardcoded Classifier #######################
    # Train the program
   # print(myClassifier.fit(dataTrain, targetTest))

    # Send the test data and get a prediction
    #myClassifier.predict(dataTest)

    # Print out the accuracy
   # print(myClassifier.calcAccuracy(targetTest), "% Correct")

    ####################################################################
    ####################################################################
    ####################################################################
    ############################ Decision Tree #########################

    # decisionTree = DecisionTree()
    # decisionTree.makeTree((iris.data,))


if __name__ == '__main__':
    main()

