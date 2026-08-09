import os
import time


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def load_print(statement: str):
    time.sleep(1)
    print(statement)
    time.sleep(2)


def function_header(statement: str):
    time.sleep(1)
    print(f"----{statement}----")
    time.sleep(1)
