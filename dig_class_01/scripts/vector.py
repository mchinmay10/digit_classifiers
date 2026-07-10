"""
Contains the following functions:
1. zeros
2. ones
3. get_address_space
4. raw_copy
5. copy_scratch
6. copy_scratch_v2
7. add
8. subtract
9. scaler_multiply
10. dot
11. magnitude
12. distance
"""

import time
import math
from data_helper import vector_length


# creates a list containing 'n' zeros
def zeros(n):
    zeros_list = []
    for _ in range(n):
        zeros_list.append(0)
    return zeros_list


# creates a list containing 'n' ones
def ones(n):
    ones_list = []
    for _ in range(n):
        ones_list.append(1)
    return ones_list


# checking address space for two variables
def get_address_space(u, v):
    print("Check address space for u and v")
    time.sleep(2)
    address_hex_u = hex(id(u))
    address_hex_v = hex(id(v))
    print(f"Address of u: {address_hex_u}")
    print(f"Address of v: {address_hex_v}")
    return address_hex_u, address_hex_v


# what happens when we do u = v ?
def raw_copy():
    u = [10, 7, 11]
    print(f"u: {u}")
    v = u
    print("Performing v = u operation")
    time.sleep(2)
    print(f"v: {v}")
    address_u, address_v = get_address_space(u, v)
    if address_u == address_v:
        print("v is just another name for the same variable u")
        print("Modifying v...")
        time.sleep(2)
        v[0] = 0
        print(f"v: {v}")
        print("Verifying u...")
        time.sleep(2)
        print(f"u: {u}")
        mod_addr_u, mod_addr_v = get_address_space(u, v)


# make another copy of the list
def copy_scratch(u):
    v = []
    for item in u:
        v.append(item)
    return v


# can we implement copy scratch as follows
def copy_scratch_v2(u):
    return u


# adding two vectors from scratch
def add(u, v):
    if vector_length(u) == vector_length(v):
        sum = []
        for x, y in zip(u, v):
            sum.append(x + y)
            return sum
    else:
        print(f"add(u, v) is invalid as {vector_length(u)} != {vector_length(v)}")
        return 1


# subtracting a vector from another from scratch
def substract(u, v):
    if vector_length(u) == vector_length(v):
        sub = []
        for x, y in zip(u, v):
            sub.append(x - y)
            return sub
    else:
        print(f"subtract(u, v) is invalid as {vector_length(u)} != {vector_length(v)}")
        return -1


# multiplying the vector 'v' by a scalar quantity 'c'
def scalar_multiply(v, c):
    cv = []
    for value in v:
        cv.append(c * value)
    return cv


# implementing the dot product between two vectors from scratch
def dot(u, v):
    if vector_length(u) == vector_length(v):
        dot_ = 0
        for x, y in zip(u, v):
            dot_ += x * y
        return dot_
    else:
        print(f"dot(u, v) is invalid as {vector_length(u)} != {vector_length(v)}")


# implementing magnitude of a vector from scratch
def magnitude(v):
    mag = 0
    for item in v:
        mag += item**2

    return math.sqrt(mag)


# implementing the distance formula
def distance(u, v):
    dist_sqrd = 0
    if vector_length(u) == vector_length(v):
        for x, y in zip(u, v):
            dist_sqrd += (x - y) ** 2

        dist = math.sqrt(dist_sqrd)
        return dist
    else:
        print(f"distance(u, v) is invalid as {vector_length(u)} != {vector_length(v)}")
        return 0


# Test cases
def zeros_test():
    print("Executing test cases for zeros function...")
    time.sleep(3)
    print(f"zeros(5): {zeros(5)}")


def ones_test():
    print("Executing test cases for ones function...")
    time.sleep(3)
    print(f"ones(5): {ones(5)}")


def copy_scratch_test():
    print("Executing test cases for copy scratch function...")
    time.sleep(3)
    u = [1, 2, 3]
    print(f"u: {u}")
    print("Performing copy_scratch(u)...")
    time.sleep(3)
    v = copy_scratch(u)
    print(f"v: {v}")
    print("Printing addresses of u and v...")
    time.sleep(3)
    addr_u, addr_v = get_address_space(u, v)


def copy_scratch_v2_test():
    print("Executing test cases for copy scratch v2 function...")
    time.sleep(3)
    u = [1, 2, 3]
    print(f"u: {u}")
    print("Performing copy_scratch_v2(u)...")
    time.sleep(3)
    v = copy_scratch_v2(u)
    print(f"v: {v}")
    time.sleep(3)
    addr_u, addr_v = get_address_space(u, v)
    if addr_u == addr_v:
        print(f"Copy scratch v2 fails the purpose!")


def dot_test():
    print("Executing test cases for dot function...")
    time.sleep(3)
    u = [1, 2, 3]
    v = [4, 5, 6]
    dot_prod = dot(u, v)
    print(f"dot(u, v) = {dot_prod}")


def distance_test():
    print("Executing test cases for distance function...")
    time.sleep(3)
    print(f"distance([3, 5, 6], [3, 5, 6]) = {distance([3, 5, 6], [3, 5, 6])}")
    print(f"distance([1, 6, 8], [3, 5, 6]) = {distance([1, 6, 8], [3, 5, 6])}")
    if distance([1, 2], [3, 4, 6]):
        print(f"distance([1, 2] , [3, 4, 6]) = {distance([1, 2], [3, 4, 6])}")


# zeros_test()
# ones_test()
# raw_copy()
# copy_scratch_test()
# copy_scratch_v2_test()
# dot_test()
# distance_test()
