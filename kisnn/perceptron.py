from utils.activation_functions import ActivationFunctions, ActivationType
from random import random

class Perceptron:
    """The most basic part of a neural network.

    Receives x inputs, calculates their weighted sum (with a bias),
    which passes through an activation function to generate an output.
    """
    def __init__(self, inputs_number, weights = None, bias = None):
        if weights is not None:
            if not isinstance(weights, list):
                raise TypeError("Weights must be passed as a list of weights")
            if len(weights) != inputs_number:
                raise ValueError("Weights has %d values, expected %d "
                                  % (len(weights), inputs_number))

        self._inputs_num = int(inputs_number)

        if weights is not None:
            self._weights = weights
        else:
            # If weights were not passed as an argument, create them
            self._weights = []
            for i in range(inputs_number):
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

    def output(self, inputs, activation_type = ActivationType.SIGN):
        """Generates the output.

        Passes the weighted sum of this perceptron's inputs through an
        activation function which generates the output.

        Parameters:
        list inputs: list containing the inputs
        IntEnum ActivationType: activation function type (default: sign(x))

        Returns:
        float output
        """
        return ActivationFunctions.activate(self.weighted_sum(inputs),
                                            activation_type)

    def learn(self, inputs,
              answer, learning_rate,
              activation_type = ActivationType.SIGN):
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
        IntEnum ActivationType: activation function type (default: sign(x))
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
    inp = [[1,1], [1,0], [0,1], [0,0]]
    ans = [    1,    -1,    -1,    -1]
    lr = 1/9

    p = Perceptron( len(inp[0]) )
    squared_error = 1
    prev_sq_err = 0
    i = 0

    while squared_error > .1 and i < 1000:
        guess = []
        for x in range(len(inp)):
            guess.append(p.output(inp[x]))

        wsum = []
        for x in range(len(inp)):
            wsum.append(p.weighted_sum(inp[x]))

        error = []
        for x in range(len(ans)):
            error.append(ans[x] - guess[x])

        prev_sq_err = squared_error
        squared_error = 0
        for x in range(len(error)):
            squared_error += error[x]*error[x]

        squared_error = round(squared_error, 16)

        print("Iteration %d" % i)
        print("Learning Rate: %.3f\n" % lr)
        print(p)
        print()

        print("Inputs:   ", end="")
        for x in range(len(inp)):
            print("{0}".format(inp[x]), end = " ")
        print()

        print("WSum:     ", end="")
        for x in range(len(inp)):
            spacing = len(("{0}".format(inp[x])))
            print(("{0:"+str(spacing)+"."+str(spacing-3)+"f}").format(wsum[x]),
            end = " ")
        print()

        print("Output:   ", end="")
        for x in range(len(inp)):
            spacing = len(("{0}".format(inp[x])))
            print(("{0:"+str(spacing)+"."+str(spacing-3)+"f}").format(guess[x]),
            end = " ")
        print()

        print("Expected: ", end="")
        for x in range(len(inp)):
            spacing = len(("{0}".format(inp[x])))
            print(("{0:"+str(spacing)+"."+str(spacing-3)+"f}").format(ans[x]),
                   end = " ")
        print()

        print("Error:    ", end="")
        for x in range(len(inp)):
            spacing = len(("{0}".format(inp[x])))
            print(("{0:"+str(spacing)+"."+str(spacing-3)+"f}").format(error[x]),
                   end = " ")
        print("\n")

        spacing = len(("{0}".format(inp[0])))
        print("Sq Error: " + ("{0:"+str(spacing)+"."+str(spacing-3)+"f}")
              .format(squared_error))

        if i != 0:
            print("Previous: ", end="")
            spacing = len(("{0}".format(inp[0])))
            print(("{0:"+str(spacing)+"."+str(spacing-3)+"f}")
                  .format(prev_sq_err))

            print("Gain:     ", end="")
            spacing = len(("{0}".format(inp[0])))
            print(("{0:"+str(spacing)+"."+str(spacing-3)+"f}")
                  .format(prev_sq_err-squared_error))

        print("------")

        for x in range(len(inp)):
            p.learn(inp[x], ans[x], lr)

        i+=1
