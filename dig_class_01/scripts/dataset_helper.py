"""
Contains the following functions:
1. data_label_split
2. get_label_frequencies
3. get_dataset_pixel_info
4. get_digit_pixel_info
5. inspect_dataset
6. avg_num_of_nonzero_pixel_per_dig
"""

import inspect
import executing
import pandas as pd
import numpy as np

from data_helper import count_nonzero

train, valid, test = pd.read_pickle("../data/mnist.pkl")


def data_label_split(split):
    x_split, y_split = split

    return x_split, y_split


def get_label_frequencies(y_split):
    vals, counts = np.unique(y_split, return_counts=True)
    raw_dict = dict(zip(vals, counts))
    clean_dict = {int(k): int(v) for k, v in raw_dict.items()}
    return clean_dict


def get_dataset_pixel_info(x_split):
    MIN_ELE = 255
    MAX_ELE = 0
    row_avg = 0

    for row in x_split:
        row_min = np.min(row)
        row_max = np.max(row)
        row_avg += np.mean(row)
        if MIN_ELE > row_min:
            MIN_ELE = row_min
        if MAX_ELE < row_max:
            MAX_ELE = row_max

    AVG_VAL = row_avg / len(x_split)

    return (MIN_ELE, MAX_ELE, AVG_VAL)


def get_digit_pixel_info(split):
    x_split, y_split = data_label_split(split)
    digit_pixel_info = {}
    data_len = len(x_split)

    for i in range(data_len):
        key = int(y_split[i])
        val = count_nonzero(x_split[i])
        if key in digit_pixel_info:
            digit_pixel_info[key] += val
        else:
            digit_pixel_info[key] = val

    freq_dict = get_label_frequencies(y_split)
    for k in digit_pixel_info.keys() and freq_dict.keys():
        digit_pixel_info[k] /= freq_dict[k]

    return digit_pixel_info


def inspect_dataset(split):
    frame = inspect.currentframe().f_back
    source_node = executing.Source.executing(frame).node
    argument_name = source_node.args[0].id
    x_split, y_split = data_label_split(split)
    min_pixel_val, max_pixel_val, avg_pixel_val = get_dataset_pixel_info(x_split)
    print(f"Dataset name: {argument_name}")
    print(f"Number of images: {len(split[0])}")
    print(f"Label frequencies: {get_label_frequencies(y_split)}")
    print(
        f"Min pixel value: {min_pixel_val}\nMax pixel value: {max_pixel_val}\nAverage pixel value: {avg_pixel_val}"
    )
    print("\n")


def avg_num_of_nonzero_pixel_per_dig(split):
    dig_pixel_info = get_digit_pixel_info(split)
    frame = inspect.currentframe().f_back
    source_node = executing.Source.executing(frame).node
    argument_name = source_node.args[0].id
    print(
        f"Displaying average number of non zero pixels per digit in the dataset: {argument_name}"
    )
    for key, value in dig_pixel_info.items():
        print(f"{key} : {value}")


# inspect_dataset(train)
# inspect_dataset(valid)
# inspect_dataset(test)
if __name__ == "__main__":
    avg_num_of_nonzero_pixel_per_dig(train)
    avg_num_of_nonzero_pixel_per_dig(valid)
    avg_num_of_nonzero_pixel_per_dig(test)
