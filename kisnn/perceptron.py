from utils.activation_functions import ActivationFunctions, ActivationType
from random import random

class Perceptron:
    """The most basic part of a neural network.

    Receives x inputs, calculates their weighted sum (with a bias),
    which passes through an activation function to generate an output.
    """
    def __init__(self, inputs_number, weights = None, bias = None):
        self._inputs_num = int(inputs_number)
        self._weights = []

        if weights is not None:
            # If weights were passed as an argument
            if not isinstance(weights, (list, int, float)):
                # If it's not a list or a number, raise an error
                raise TypeError("Weights must be either a list or a number")

            if isinstance(weights, list):
                # If it's a list, check if its lenght is valid
                if len(weights) != int(inputs_number):
                    raise ValueError("Weights has %d values, expected %d "
                                    % (len(weights), int(inputs_number)))
                # Lenght is valid, we're set
                self._weights = weights
            else:
                # If it's a number, create a list of weights with it
                self._weights = [float(weights)] * int(inputs_number)

        else:
            # If weights were not passed as an argument, create them
            self._weights = []
            for i in range(int(inputs_number)):
                self._weights.append(random()*2-1)

        # Add the Bias
        if bias is not None:
            self._weights.append(float(bias))
        else:
            self._weights.append(random()*2-1)

    def __str__(self):
        weights = self._weights[:-1]
        bias = self._weights[-1:]
        return "Weight(s): [" +                                          \
               ( "] [".join(["%5.3f"]*len(weights)) % tuple(weights) ) + \
               "]" +                                                     \
                                                                         \
               "\nBias:      [" +                                        \
               ( "".join(["%5.3f"]*len(bias)) % tuple(bias) ) +          \
               "]"

    def weighted_sum(self, inputs):
        """Calculates the weighted sum.

        Uses the formula:
        WSum = ∑(xi * wi) + bias
        where xi = Input i,
              wi = Weight i.

        Parameters:
        list inputs: list containing the inputs

        Returns:
        float wsum
        """
        if not isinstance(inputs, list):
            raise TypeError("Inputs must be a list")
        if len(inputs) != self._inputs_num:
            raise ValueError("Inputs has %d values, expected %d"
                            % (len(inputs), self._inputs_num))

        wsum = 0
        for i in range(self._inputs_num):
            wsum += float(inputs[i]) * float(self._weights[i])

        # Bias
        wsum += float(self._weights[-1])
        return round(wsum, 10) # Rounded to the 10th digit to return
                               # "human-friendly" numbers

    def output(self, inputs, activation_type = ActivationType.STEP):
        """Generates the output.

        Passes the weighted sum of this perceptron's inputs through an
        activation function which generates the output.

        Parameters:
        list inputs: list containing the inputs
        IntEnum ActivationType: activation function type (default: step(x))

        Returns:
        float output
        """
        return ActivationFunctions.activate(self.weighted_sum(inputs),
                                            activation_type)

    def learn(self, inputs,
              answer, learning_rate,
              activation_type = ActivationType.STEP):
        """Supervisioned learning algorithm.

        Given the answer, trains this perceptron to evaluate a certain input
        correctly.
        The perceptron takes a guess on the answer, then the error is
        calculated (error = answer - guess).
        Then, adjusts the perceptron's weights and bias according to error and
        a learning rate.

        Parameters:
        list inputs: list containing the inputs
        float answer: expected result
        float learning_rate: rate that determines how much the weights
                             adjusting modifies the weights and bias
                             (generally less than .5)
        IntEnum ActivationType: activation function type (default: step(x))
        """
        guess = self.output(inputs, activation_type)
        error = float(answer) - float(guess)
        self.adjust_weights(error, inputs, learning_rate)

    def adjust_weights(self, error, inputs, learning_rate):
        """Weight adjusting algorithm.

        Uses the formulas:
        ΔW = Error * Input
        New_Weight = Old_Weight + ΔW * Learning_Rate

        Parameters:
        float error: error between expected answer and perceptron's guess
        list inputs: list containing the inputs
        float learning_rate: rate which determines how much the weights
                             are modified (generally less than .5)
        """
        # Adjust the weights
        for i in range(self._inputs_num):
            delta_weight = float(error) * float(inputs[i])
            self._weights[i] += delta_weight * float(learning_rate)

        # Adjust the bias
        self._weights[-1] += float(error) * float(learning_rate)

# Testing
if __name__ == "__main__":
    # Instantiation tests

    # Only inputs_number
    p = Perceptron(2)
    print(p)
    print("---")

    # inputs_number and a float weight, random bias
    p = Perceptron(5, .67)
    print(p)
    print("---")

    # inputs_number and an int weight, random bias
    p = Perceptron(4, 3)
    print(p)
    print("---")

    # inputs_number and a list of weights, random bias
    p = Perceptron(5, [.5, -.6, -7, 8, 0])
    print(p)
    print("---")

    # inputs_number and a float weight, int bias
    p = Perceptron(5, -.4, 1)
    print(p)
    print("---")

    # inputs_number and a float weight, float bias
    p = Perceptron(5, -7.77, .33)
    print(p)
    print("---")

    # inputs_number and an int weight, int bias
    p = Perceptron(4, 0, -3)
    print(p)
    print("---")

    # inputs_number and an int weight, float bias
    p = Perceptron(4, 3, 4/7)
    print(p)
    print("---")

    # inputs_number and a list of weights, int bias
    p = Perceptron(3, [.5, -.6, 0], 0)
    print(p)
    print("---")

    # inputs_number and a list of weights, float bias
    p = Perceptron(4, [-.6, -7, 8, 0], -1.17)
    print(p)
    print("---")
