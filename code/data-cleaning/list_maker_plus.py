names = []

try:
    while True:
        input = input()
        print(input)
        # names = str(input).split("+")
except EOFError:
    pass

print(names)


def f(s):
    lst = s.split(" + ")
    names = []
    for item in lst:
        names.append("hrsd01__" + item)
    print(names)


def z(s):
    lst = s.split(" + ")
    names = []
    for item in lst:
        names.append(item)
    print(names)