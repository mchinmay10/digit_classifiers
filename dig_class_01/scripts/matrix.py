import time
from data_helper import vector_length
from vector import dot


# implementing the shpe() of a matrix froms scratch:
def mat_shape(matrix):
    return (vector_length(matrix), vector_length(matrix[0]))


# implementing the matrix transpose function from scratch:
def mat_transpose(matrix):
    matrix_t = []
    temp_row = []
    row, col = mat_shape(matrix)
    for j in range(col):
        for i in range(row):
            temp_row.append(matrix[i][j])
        matrix_t.append(temp_row)
        temp_row = []

    return matrix_t


# check if the matrix multiplication is valid or not:
def isValidMatrixMul(A, B):
    _, col_A = mat_shape(A)
    row_B, _ = mat_shape(B)
    if col_A == row_B:
        return True
    else:
        return False


# implementing the matrix multiplication function from scratch:
def mat_mul(A, B):
    prod_mat = []
    temp_row = []
    if isValidMatrixMul(A, B):
        row_A, _ = mat_shape(A)
        _, col_B = mat_shape(B)
        B_t = mat_transpose(B)
        for i in range(row_A):
            for j in range(col_B):
                temp_row.append(dot(A[i], B_t[j]))
            prod_mat.append(temp_row)
            temp_row = []
        return prod_mat
    else:
        return "Matrices' dimensions does not match conditions for multiplying!"


# calculate dim of matrix to be multiplied with input layer
def dim_of_matrix(input_dim, output_dim):
    input_row, input_col = input_dim
    output_row, output_col = output_dim
    if input_row == output_row:
        return (input_col, output_col)
    else:
        return "Invalid Matrix Multiplication output"


# Test cases
def mat_shape_test():
    print("Executing test cases for matrix shape function...")
    time.sleep(3)
    print(
        f"mat_shape([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]): {mat_shape([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])}"
    )


def mat_transpose_test():
    print("Executing test cases for matrix transpose function...")
    time.sleep(3)
    print(
        f"mat_transpose([[1, 2, 3], [4, 5, 6]]): {mat_transpose([[1, 2, 3], [4, 5, 6]])}"
    )


def mat_mul_test():
    print("Executing test cases for matrix multiplication function...")
    time.sleep(3)
    # [[1, 2], [3, 4]] * [[1, 2], [3, 4]] = [[7, 10], [15, 22]]
    print(
        f"mat_mul([[1, 2], [3, 4]], [[1, 2], [3, 4]]): {mat_mul([[1, 2], [3, 4]], [[1, 2], [3, 4]])}"
    )
    print(
        f"mat_mul([[1, 2, 3], [4, 5, 6]], [[1, 2], [3, 4], [5, 6]]): {mat_mul([[1, 2, 3], [4, 5, 6]], [[1, 2], [3, 4], [5, 6]])}"
    )
    print(
        f"mat_mul([[1, 2, 3], [4, 5, 6]], [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]): {mat_mul([[1, 2, 3], [4, 5, 6]], [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]])}"
    )


def dim_of_matrix_test():
    print(
        "Executing test cases for calculating the dimension of multiplying matrix to the input layer..."
    )
    time.sleep(3)
    print(
        f"Dimension of required matrix for input: (1, 784) and output: (1, 10) is {dim_of_matrix((1, 784), (1, 10))}"
    )
    print(
        f"Dimension of required matrix for input: (1, 784) and output: (10, 1) is {dim_of_matrix((1, 784), (10, 1))}"
    )


if __name__ == "__main__":
    print(f"----Running test cases for matrix.py file----")
    mat_shape_test()
    mat_transpose_test()
    mat_mul_test()
    dim_of_matrix_test()
