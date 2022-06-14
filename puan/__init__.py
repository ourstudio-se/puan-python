import typing
from dataclasses import dataclass

@dataclass(frozen=True, eq=False)
class variable(object):
    
    """
        The variable class is a central key in all Puan packages. It consists of an id, data type (dtype) and if it is virtual or not.
        A virtual variable is a variable that has been created along some reduction and is not (necessary) interesting for the user.
    """

    id: str
    dtype: typing.Union[bool, int]
    virtual: bool

    def __hash__(self):
        return hash(self.id)

    def __lt__(self, other):
        return self.id < other.id

    def __eq__(self, o):
        return self.id == o.id

    @staticmethod
    def construct(*variable_ids: typing.List[str], dtype: typing.Union[bool, int] = bool, virtual: bool = False) -> typing.List["variable"]:
        return list(map(lambda v: variable(v, dtype, virtual), variable_ids))
