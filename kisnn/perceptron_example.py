from perceptron import Perceptron
from utils.activation_functions import ActivationType

print("Trainig perceptron to solve logical AND\n")

inputs =          [[1,1], [1,0], [0,1], [0,0]]
answer =          [    1,     0,     0,     0]

max_iterations =  100                 # Caution when modifying this
learning_rate =   0.05                # Try setting to .1, .01, .001, or 5
activation_type = ActivationType.STEP # IDENTITY, SIGN, STEP, or SIGMOID

p = Perceptron( len(inputs[0]) )
squared_error = 1
prev_sq_err = 0
i = 0

while squared_error > .1 and i <= max_iterations:
    guess = []
    for x in range(len(inputs)):
        guess.append(p.output(inputs[x], activation_type))

    wsum = []
    for x in range(len(inputs)):
        wsum.append(p.weighted_sum(inputs[x]))

    error = []
    for x in range(len(inputs)):
        error.append(answer[x] - guess[x])

    prev_sq_err = squared_error
    squared_error = 0
    for x in range(len(inputs)):
        squared_error += error[x]*error[x]

    squared_error = round(squared_error, 16)

    print("Iteration %d" % i)
    print("Learning Rate: %.3f\n" % learning_rate)
    print(p)
    print()

    to_print = "Inputs:    "
    for x in range(len(inputs)):
        to_print += ("{0}".format(str(inputs[x]))) + " "
    print(to_print)

    to_print = "WSum:      "
    for x in range(len(inputs)):
        spacing = len(("{0}".format(str(inputs[x]))))
        to_print += ("{0:"+ str(spacing) + "." + str(spacing-3) + "f}") \
                    .format(wsum[x]) + " "
    print(to_print)

    to_print = "Output:    "
    for x in range(len(inputs)):
        spacing = len(("{0}".format(str(inputs[x]))))
        to_print += ("{0:"+str(spacing)+"."+str(spacing-3)+"f}") \
                    .format(guess[x]) + " "
    print(to_print)

    to_print = "Expected:  "
    for x in range(len(inputs)):
        spacing = len(("{0}".format(str(inputs[x]))))
        to_print += ("{0:"+str(spacing)+"."+str(spacing-3)+"f}") \
                    .format(answer[x]) + " "
    print (to_print)

    to_print = "Error:     "
    for x in range(len(inputs)):
        spacing = len(("{0}".format(str(inputs[x]))))
        to_print += ("{0:"+str(spacing)+"."+str(spacing-3)+"f}") \
                    .format(error[x]) + " "
    print(to_print + "\n")

    spacing = len(("{0}".format(str(inputs[0]))))
    print("Sq Error:  " + ("{0:"+str(spacing)+"."+str(spacing-3)+"f}") \
                          .format(squared_error))

    if i != 0:
        to_print = "Previous:  "
        spacing = len(("{0}".format(str(inputs[0]))))
        to_print += ("{0:"+str(spacing)+"."+str(spacing-3)+"f}") \
                    .format(prev_sq_err)
        print(to_print)

        to_print = "Gain:      "
        spacing = len(("{0}".format(str(inputs[0]))))
        to_print += ("{0:"+str(spacing)+"."+str(spacing-3)+"f}") \
                    .format(prev_sq_err-squared_error)
        print(to_print)

    print("------")

    for x in range(len(inputs)):
        p.learn(inputs[x], answer[x], learning_rate, activation_type)

    i += 1
