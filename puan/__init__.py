import typing

# import puan.npufunc
class variable(object):
    
    """
        The variable class is a central key in all Puan packages. It consists of an id, data type (dtype) and if it is virtual or not.
        A virtual variable is a variable that has been created along some reduction and is not (necessary) interesting for the user.
    """

    def __init__(self, id: str, dtype: typing.Union[bool, int], virtual: bool = False):
        self.id = id
        self.dtype = dtype
        self.virtual = virtual

    def __hash__(self):
        return hash(self.id)

    def __lt__(self, other):
        return self.id < other.id

    def __eq__(self, o):
        return self.id == o.id

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"'{self.id}': {self.dtype} {'(virtual)' if self.virtual else ''}"
