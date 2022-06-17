import typing
import maz
import itertools
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
    def from_strings(*variables: typing.List[str], dtype_default: typing.Union[bool, int] = bool, virtual_default: bool = False) -> typing.List["variable"]:

        """
            Returns a list of puan.variable from a list of strings (id's)

            Examples
            --------
                >>> variable.from_strings("a","b")
                [variable(id="a",dtype=bool,virtual=False),variable(id="b",dtype=bool,virtual=False)]

                >>> variable.from_strings("a","b", dtype_default=int, virtual_default=True)
                [variable(id="a",dtype=bool,virtual=False),variable(id="b",dtype=bool,virtual=False)]

            Returns
            -------
                out : typing.List[variable]
        """

        return list(map(lambda v: variable(v, dtype, virtual), variables))

    @staticmethod
    def from_mixed(*variables: typing.List[typing.Union[str, int, tuple, list, "variable"]], dtype_default : typing.Union[bool, int] = bool, virtual_default: bool = False) -> typing.List["variable"]:
        
        """
            Returns a list of puan.variable from a list of mixed data type.

            Notes
            -----
            Every item in *variables that is not an instance of `variable` will be converted to a string and used as an id.


            Examples
            --------
                >>> variable.from_mixed("a",4,("b","c"),variable("x",int,True))
                [variable(id="a",dtype=bool,virtual=False),variable(id="4",dtype=bool,virtual=False),variable(id="('b','c')",dtype=bool,virtual=False),variable("x",int,True)]

                >>> variable.from_mixed("a",4,variable("x",int,True), dtype_default=int, virtual=True)
                [variable(id="a",dtype=int,virtual=True),variable(id="4",dtype=int,virtual=True),variable("x",int,True)]

            Returns
            -------
                out : typing.List["variable"]
        """
        return list(
            itertools.chain(
                filter(
                    lambda v: isinstance(v, variable),
                    variables
                ),
                map(
                    lambda v: variable(
                        str(v), 
                        dtype=dtype_default, 
                        virtual=virtual_default,
                    ),
                    filter(
                        lambda v: not isinstance(v, variable),
                        variables
                    )
                )
            )
        )
