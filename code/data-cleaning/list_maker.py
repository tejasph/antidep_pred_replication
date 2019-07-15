"""
Allows you to paste strings that are separated by a newline. Spits out the python list representation of it.
"""

names = []

try:
    while True:
        names.append(input())
except EOFError:
    pass

print(names)

