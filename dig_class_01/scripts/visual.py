import os
import time


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def load_print(statement: str):
    time.sleep(1)
    print(statement)
    time.sleep(2)


def border_print_v1(statement: str):
    top_border = "⎽" * len(statement)
    bottom_border = "‾" * len(statement)
    time.sleep(1)
    print(f"{top_border}\n{statement}\n{bottom_border}")
    time.sleep(1)


def function_header(statement: str):
    time.sleep(1)
    print(f"----{statement}----")
    time.sleep(1)


# Test cases
def border_print_v1_test():
    function_header("Test cases for the border_print_v1 function")
    border_print_v1("Hello World")


if __name__ == "__main__":
    border_print_v1_test()
