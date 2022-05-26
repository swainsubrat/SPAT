import sys
print(sys.path)
# sys.path.append("./")
print(sys.path)
from ..helper import hello

hello()


def bye():
    print("bye")