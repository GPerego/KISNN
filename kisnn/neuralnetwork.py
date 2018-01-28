from perceptron import Perceptron

class NeuralNetwork:

    def __init__(self, inputs, hidden, outputs):
        self._inputs = []
        self._hidden = []
        self._outputs = []

        for i in range(int(inputs)):
            self._inputs.append(Perceptron(1))

        for i in range(int(hidden)):
            self._hidden.append(Perceptron(inputs))

        for i in range(int(outputs)):
            self._outputs.append(Perceptron(hidden))

    def __str__(self):
        to_print = "Inputs:\n\n"
        for i in range(len(self._inputs)):
            to_print += "Input " + str(i+1) + ":\n"
            to_print += str(self._inputs[i]) + "\n"
            if i+1 != len(self._inputs):
                to_print += "\n"
        to_print += "------\n"

        to_print += "Hidden:\n\n"
        for i in range(len(self._hidden)):
            to_print += "Hidden " + str(i+1) + ":\n"
            to_print += str(self._hidden[i]) + "\n"
            if i+1 != len(self._hidden):
                to_print += "\n"
        to_print += "------\n"

        to_print += "Outputs: \n\n"
        for i in range(len(self._outputs)):
            to_print += "Output " + str(i+1) + ":\n"
            to_print += str(self._outputs[i]) + "\n"
            if i+1 != len(self._outputs):
                to_print += "\n"
        to_print += "-~~~~-"

        return to_print

if __name__ == "__main__":
    nn = NeuralNetwork(4,4,1)
    print(nn)
