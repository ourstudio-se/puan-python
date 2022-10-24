import json
import numpy
import typing
import maz
import itertools
import dataclasses

default_min_int     : int        = numpy.iinfo(numpy.int16).min
default_max_int     : int        = numpy.iinfo(numpy.int16).max
default_int_bounds  : (int, int) = (default_min_int, default_max_int)

def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)

class StatementInterface:

    def to_short(self) -> tuple:
        
        """Short statement has (id, sign, propositions, value, bounds)"""
        raise NotImplementedError()

    def to_json(self) -> dict:

        """json representation of class"""
        raise NotImplementedError()

    def to_dict() -> dict:

        """dict representation of class"""
        raise NotImplementedError()

    @staticmethod
    def from_json(self, class_map) -> typing.Any:

        """Convert from json into class object. Class map is a list of other classes maybe related to this class"""
        raise NotImplementedError()

@dataclasses.dataclass
class variable(StatementInterface):

    """
        The variable class is a central key in all Puan packages. It consists of an id, data type (dtype) and if it is virtual or not.
        A virtual variable is a variable that has been created along some reduction and is not (necessary) interesting for the user.
    """

    id: str
    bounds: typing.Tuple[int, int] = (0, 1)

    def __init__(self, id: str, bounds: typing.Tuple[int, int] = (0, 1), dtype: str = None):
        self.id = id
        self.bounds = tuple(bounds)
        if dtype is not None:
            self.bounds = {"int": default_int_bounds, "bool": (0, 1)}.get(dtype, (0, 1))

    def __hash__(self):
        return hash(self.id)

    def __lt__(self, other):
        if not hasattr(other, "id"):
            return True
        return self.id < other.id

    def __eq__(self, other):
        return (self.id == getattr(other, "id", other)) and (self.bounds == getattr(other,"bounds",(0,1)))

    def to_json(self):
        d = {'id': self.id}
        if self.bounds != (0, 1):
            d['bounds'] = {'min': self.bounds[0], 'max': self.bounds[1]}
        return d

    def to_short(self):
        return (self.id, 1, [], 0, self.bounds)

    def to_dict(self):
        return {self.id: (1, [], 0, self.bounds)}

    @staticmethod
    def support_vector_variable():
        return variable(0, dtype="int")

    @staticmethod
    def from_strings(*variables: typing.List[str], default_bounds: typing.Union[bool, int] = (0,1)) -> typing.List["variable"]:

        """
            Returns a list of puan.variable from a list of strings (id's)

            Notes
            -----
            List of variables are returned sorted

            Examples
            --------
                >>> variable.from_strings("a","b")
                [variable(id='a', bounds=(0, 1)), variable(id='b', bounds=(0, 1))]

                >>> variable.from_strings("a","b", default_bounds=(-1,10))
                [variable(id='a', bounds=(-1, 10)), variable(id='b', bounds=(-1, 10))]

            Returns
            -------
                out : typing.List[variable]
        """

        return sorted(map(lambda v: variable(v, default_bounds), variables))

    @staticmethod
    def from_mixed(*variables: typing.List[typing.Union[str, int, tuple, list, "variable"]], default_bounds : tuple = (0,1)) -> typing.List["variable"]:
        
        """
            Returns a list of puan.variable from a list of mixed data type.

            Notes
            -----
            - Every item in ``*variables`` that is not an instance of ``variable`` will be converted to a string and used as an id.
            - List of variables are returned sorted.

            Examples
            --------
                >>> variable.from_mixed("a",4,("b","c"),variable("x",(1,2)))
                [variable(id="('b', 'c')", bounds=(0, 1)), variable(id='4', bounds=(0, 1)), variable(id='a', bounds=(0, 1)), variable(id='x', bounds=(1, 2))]

                >>> variable.from_mixed("a",4,variable("x",(2,4)), default_bounds=(-1, 1))
                [variable(id='4', bounds=(-1, 1)), variable(id='a', bounds=(-1, 1)), variable(id='x', bounds=(2, 4))]

            Returns
            -------
                out : typing.List["variable"]
        """
        return sorted(
            itertools.chain(
                filter(
                    lambda v: isinstance(v, variable),
                    variables
                ),
                map(
                    lambda v: variable(
                        str(v), 
                        default_bounds
                    ),
                    filter(
                        lambda v: not isinstance(v, variable),
                        variables
                    )
                )
            )
        )

    @staticmethod
    def from_json(data: dict, class_map):
        bounds = data.get('bounds', {'min': 0, 'max': 1})
        return variable(
            id=data['id'],
            bounds=(bounds['min'], bounds['max'])
        )


class SolutionVariable(variable):

    value: int = None
    def __init__(self, id: str, bounds: typing.Tuple[int, int] = (0, 1), dtype: str = None, value: int = None):
        super().__init__(id, bounds, dtype)
        self.value = value

    def __eq__(self, other):
        return self.id == other.id and self.value == other.value

    @staticmethod
    def from_variable(variable: variable, value: int) -> "SolutionVariable":
        return SolutionVariable(variable.id, variable.bounds, value=value)