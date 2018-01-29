from enum import IntEnum
from math import exp

class ActivationType(IntEnum):
    NONE = 1
    SIGN = 2
    SIGMOID = 3

class ActivationFunctions:
    @staticmethod
    def activate(x, type):
        if(type == ActivationType.NONE):
            return x
        if(type == ActivationType.SIGN):
            return ActivationFunctions.sign(x)
        if(type == ActivationType.SIGMOID):
            return ActivationFunctions.sigmoid(x)
        else:
            raise ValueError("Invalid Activation Type")

    @staticmethod
    def sign(x):
        return 1 if x >= 0 else -1

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + exp(-x))

# Testing
if __name__ == "__main__":
    test = [0, 1.5, -.6]

    print("Activation Functions Testing")
    print()

    for x in ActivationType:
        print("Testing: " + str(x) + "\n")
        for i in range(len(test)):
            print("Test %d - Value: %.2f" % (i, test[i]))
            print("Result: " + str(ActivationFunctions.activate(test[i], x)))
            print()
        print("------")
