from enum import IntEnum

class ActivationType(IntEnum):
    SIGN = 1

class ActivationFunctions:
    @staticmethod
    def activate(x, type):
        if(type == 1):
            return ActivationFunctions.sign(x)
        else:
            raise ValueError("Invalid Activation Type")

    @staticmethod
    def sign(x):
        return 1 if x >= 0 else -1

# Testing
if __name__ == "__main__":
    print(str(ActivationFunctions.activate(-0.2, ActivationType.SIGN)))
    print(str(ActivationFunctions.activate(0, ActivationType.SIGN)))
    print(str(ActivationFunctions.activate(1, ActivationType.SIGN+1)))
