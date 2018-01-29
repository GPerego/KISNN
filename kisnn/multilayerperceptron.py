from perceptron import Perceptron

class MultilayerPerceptron:
    def __init__(self, inputs, outputs, hidden_layers, hidden_numbers):
        self._inputs = []
        self._hidden = []
        self._outputs = []
        hid_num = []

        if isinstance(hidden_numbers, list):
            if len(hidden_numbers) != int(hidden_layers):
                # hidden_numbers is a list with incorrect lenght
                raise ValueError("List of hidden layer number of " + \
                                 "perceptrons has %d values, expected %d"
                                  % (len(hidden_numbers), int(hidden_layers)))
            else:
                # hidden_numbers is a list with correct lenght
                hid_num = hidden_numbers
        else:
            # hidden_numbers isn't a list, so create a list with that value*
            #                                        *(assuming it's an int)
            hid_num = [int(hidden_numbers)] * int(hidden_layers)

        # input perceptrons with only 1 input each
        for i in range(int(inputs)):
            self._inputs.append(Perceptron(1))

        # hidden layer perceptrons
        # self._hidden is a list of a list of perceptrons, with each index
        # having hid_num[i] perceptrons
        for i in range(int(hidden_layers)):
            hidden = []
            for j in range(int(hid_num[i])):
                if i == 0:
                    # First hidden layer has perceptrons with "inputs" inputs
                    hidden.append(Perceptron(int(inputs)))
                else:
                    # Other hidden layers have perceptrons with the number of
                    # the previous hidden layer perceptrons inputs
                    hidden.append(Perceptron(hid_num[i-1]))
            self._hidden.append(hidden)

        # output layer has perceptrons with n inputs, n being the number of
        # last hidden layer's perceptrons
        for i in range(int(outputs)):
            self._outputs.append(Perceptron(hid_num[-1]))

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
            for j in range(len(self._hidden[i])):
                to_print += "Hidden[" + str(i+1) + "][" + str(j+1) + "]:\n"
                to_print += str(self._hidden[i][j]) + "\n"
                if j+1 != len(self._hidden[i]):
                    to_print += "\n"
            if i+1 != len(self._hidden):
                to_print += "\nxxxxxx\n\n"
        to_print += "------\n"

        to_print += "Outputs: \n\n"
        for i in range(len(self._outputs)):
            to_print += "Output " + str(i+1) + ":\n"
            to_print += str(self._outputs[i]) + "\n"
            if i+1 != len(self._outputs):
                to_print += "\n"

        return to_print

if __name__ == "__main__":
    print("A MLP:")
    mlp = MultilayerPerceptron(4,1,3,[5,6,2])
    print(mlp)
    print()

    print("Another MLP:")
    mlp2 = MultilayerPerceptron(3,2,4,5)
    print(mlp2)
