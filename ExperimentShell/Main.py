# Runs the main program
from HardcodedClassifier import HardcodedClassifier
from DecisionTree import DecisionTree
from KNN import KNN
from sklearn import datasets
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# starter method
def main():
    #myClassifier = HardcodedClassifier()
    KNNClassifier = KNN()
    iris = datasets.load_iris()

    # Show the data (the attributes of each instance)
    # print(iris.data)
    #
    # # Show the target values (in numeric format) of each instance
    # print(iris.target)
    #
    # # Show the actual target names that correspond to each number
    # print(iris.target_names)
    #

    dataTrain, dataTest, targetTrain, targetTest = train_test_split(iris.data, iris.target, test_size=0.15)

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

    ####################################################################
    ####################################################################
    ####################################################################
    ########################## KNN Classifier ##########################

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

if __name__ == '__main__':
    main()
