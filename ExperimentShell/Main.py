# Runs the main program
from HardcodedClassifier import HardcodedClassifier
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split


# starter method
def main():
    myClassifier = HardcodedClassifier()

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

    dataTrain, dataTest, targetTrain, targetTest = train_test_split(iris.data, iris.target, test_size=0.30)

    # Train the program
    print(myClassifier.fit(dataTrain))

    # Send the test data and get a prediction
    myClassifier.predict(dataTest)

    # Print out the accuracy
    print(myClassifier.calcAccuracy(targetTest), "% Correct")


if __name__ == '__main__':
    main()
