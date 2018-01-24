from utils.activation_functions import ActivationFunctions, ActivationType

class Perceptron:
    """The most basic part of a neural network.

    Receives x inputs, calculates their weighted sum (with a bias),
    which passes through an activation function to generate an output.
    """
    def __init__(self, inputs_number, weights = []):
        if not isinstance(weights, list):
            raise TypeError("Weights must be a list")
        if weights and len(weights) != inputs_number:
            raise ValueError("Weights has %d values, expected %d"
                            % (len(weights), inputs_number))

        self._inputs_num = int(inputs_number)
        self._weights = weights

        # If weights were not passed as an argument, create them
        if not weights:
            for i in range(inputs_number):
                self._weights.append(.1) # Arbitrarily chosen number
                                         # for initial weights

        # Add the Bias
        self._weights.append(.1) # Arbitrarily chosen number for initial bias

    def __str__(self):
        return "Number of inputs: " + str(self._inputs_num) + \
               "\nWeights: " + str(self._weights[:-1]) + \
               "\nBias: " + str(self._weights[-1:])

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
        wsum += self._weights[-1]
        return wsum

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
    inp = [[1,1], [1,0], [0,1] , [0,0]]
    ans = [1, -1, -1, -1]
    guess = []
    lr = .01
    i = 0
    p = Perceptron(2)

    def print_perceptron(p, i, inp, ans):
        guess = []

        print("Iteration %d" % i)
        print()
        print(p)

        for x in range(len(inp)):
            guess.append(p.output(inp[x]))

        print("Guess: " + str(guess))
        print("Answer: " + str(ans))
        print()

        print_out(p, inp[0], guess, ans[0])
        print_out(p, inp[1], guess, ans[1])
        print_out(p, inp[2], guess, ans[2])
        print_out(p, inp[3], guess, ans[3])
        print("---------------")

    def print_out(p, inp, guess, ans):
        print("Input: " + str(inp))
        print("Output: " + str(p.output(inp)), end="")
        print(" (Weighted Sum: %f)" % p.weighted_sum(inp))
        print("Expected: " + str(ans))
        print("Error: " + str((ans) - p.output(inp)))
        print()

    print_perceptron(p, i, inp, ans)

    while guess != ans:
        guess = []
        p.learn(inp[0], ans[0], lr)
        p.learn(inp[1], ans[1], lr)
        p.learn(inp[2], ans[2], lr)
        p.learn(inp[3], ans[3], lr)
        guess.append(p.output(inp[0]))
        guess.append(p.output(inp[1]))
        guess.append(p.output(inp[2]))
        guess.append(p.output(inp[3]))
        i+=1

        print_perceptron(p, i, inp, ans)
