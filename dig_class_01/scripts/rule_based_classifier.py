import pandas as pd
from dataset_helper import data_label_split
from data_helper import count_nonzero

train, valid, test = pd.read_pickle("../data/mnist.pkl")


def get_error_in_classification(split):
    x_split, y_split = data_label_split(split)
    correct = 0
    wrong = 0
    iters = 0
    total = len(x_split)
    for x_row, y_row in zip(x_split, y_split):
        num_nonzero = count_nonzero(x_row)
        if num_nonzero > 185:
            identified_dig = 0
        if num_nonzero < 100:
            identified_dig = 1
        if num_nonzero > 165 and num_nonzero <= 170:
            identified_dig = 2
        if num_nonzero > 162 and num_nonzero <= 165:
            identified_dig = 3
        if num_nonzero > 135 and num_nonzero <= 140:
            identified_dig = 4
        if num_nonzero > 150 and num_nonzero <= 155:
            identified_dig = 5
        if num_nonzero > 156 and num_nonzero <= 162:
            identified_dig = 6
        if num_nonzero > 125 and num_nonzero <= 135:
            identified_dig = 7
        if num_nonzero > 170 and num_nonzero <= 180:
            identified_dig = 8
        if num_nonzero > 140 and num_nonzero <= 150:
            identified_dig = 9

        if y_row == identified_dig:
            correct += 1
        else:
            wrong += 1

        iters += 1
        if iters % 1000 == 0:
            print(f"identified digit: {identified_dig}")
            print(f"actual digit: {y_row}")

    print(f"Percentage of digits identified correctly: {correct / total * 100}")
    print(f"Percentage of digits identified incorrectly: {wrong / total * 100}")
    print(f"Ratio of right is to wrong: {correct / wrong}")


# get_error_in_classification(train)
# get_error_in_classification(valid)
# get_error_in_classification(test)
