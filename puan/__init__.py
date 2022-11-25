import json
import numpy
import typing
import maz
import itertools
import dataclasses
from json import JSONEncoder

default_min_int     : int        = numpy.iinfo(numpy.int16).min
default_max_int     : int        = numpy.iinfo(numpy.int16).max
default_int_bounds  : (int, int) = (default_min_int, default_max_int)

def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)

_default.default = JSONEncoder().default
JSONEncoder.default = _default

class StatementInterface:

    def to_short(self) -> typing.Tuple[str, int, object, int, typing.Tuple[int, int]]:
        
        """Short statement has (id, sign, propositions, value, bounds)"""
        raise NotImplementedError()

    def to_json(self) -> typing.Dict[str, typing.Any]:

        """json representation of class"""
        raise NotImplementedError()

    @staticmethod
    def from_json(self, class_map) -> typing.Any:

        """Convert from json into class object. Class map is a list of other classes maybe related to this class"""
        raise NotImplementedError()

@dataclasses.dataclass
class Bounds:
    """
        The :class:`Bounds` class defines lower and upper bounds of a :class:`variable`

        Methods
        -------
        as_tuple
    """

    lower: int = default_min_int
    upper: int = default_max_int

    def __init__(self, lower: int, upper: int):
        if lower > upper:
            raise ValueError(f"upper bound must be higher than lower bound, got: ({lower}, {upper})")
        self.lower = lower
        self.upper = upper

    def __iter__(self):
        return iter([self.lower, self.upper])

    def as_tuple(self) -> typing.Tuple[int, int]:
        """
            Bounds as tuple

            Returns
            -------
                out : Tuple[int, int]
        """
        return (self.lower, self.upper)

@dataclasses.dataclass
class variable(StatementInterface):

    """
        The :class:`variable` class is a central key in all Puan packages. It consists of an id, :class:`Bounds` and data type (dtype).

        Methods
        -------
        to_json
        to_short
        to_dict
        support_vector_variable
        from_strings
        from_mixed
        from_json
    """

    id: str
    bounds: Bounds

    def __init__(self, id: str, bounds: typing.Tuple[int, int] = None, dtype: str = None):
        self.id = id
        if dtype is not None:
            if bounds is not None:
                if (dtype == "bool") != ((dtype == "bool") and (bounds == (0, 1))):
                    raise ValueError("Dtype is bool thus bounds must be (0, 1), got: {}".format(bounds))
            else:
                bounds = {"int": default_int_bounds, "bool": (0, 1)}.get(dtype, (0, 1))
        elif bounds is None:
            bounds = (0, 1)
        self.bounds = Bounds(*bounds)

    def __hash__(self):
        return hash(self.id)

    def __lt__(self, other):
        return self.id < other.id

    def __eq__(self, other):
        return self.id == getattr(other, "id", other)

    def to_json(self) -> typing.Dict[str, typing.Any]:
        d = dataclasses.asdict(self)
        if self.bounds.as_tuple() == (0,1):
            del d['bounds']
        return d

    def to_short(self) -> typing.Tuple[str, int, typing.List, int, typing.Tuple[int, int]]:
        return (self.id, 1, [], 0, self.bounds.as_tuple())

    @staticmethod
    def support_vector_variable() -> "variable":
        """
            Default support vector variable settings.

            Returns
            -------
                out : :class:`variable`
        """
        return variable(0, dtype="int")

    @staticmethod
    def from_strings(*variables: typing.List[str], default_bounds: typing.Tuple[int, int] = (0,1)) -> typing.List["variable"]:

        """
            Returns a list of :class:`variable` from a list of strings (ids)

            Parameters
            ----------
                variables : A list of strings (ids)
                default_bounds : Tuple of ints (lower bound, upper bound)

            Notes
            -----
            List of variables are returned sorted

            Examples
            --------
                >>> variable.from_strings("a","b")
                [variable(id='a', bounds=Bounds(lower=0, upper=1)), variable(id='b', bounds=Bounds(lower=0, upper=1))]

                >>> variable.from_strings("a","b", default_bounds=(-1,10))
                [variable(id='a', bounds=Bounds(lower=-1, upper=10)), variable(id='b', bounds=Bounds(lower=-1, upper=10))]

            Returns
            -------
                out : List[:class:`variable`]
        """

        return sorted(map(lambda v: variable(v, default_bounds), variables))

    @staticmethod
    def from_mixed(*variables: typing.List[typing.Union[str, int, tuple, list, "variable"]], default_bounds : typing.Tuple[int, int] = (0,1)) -> typing.List["variable"]:
        
        """
            Returns a list of :class:`variable` from a list of mixed data type.

            Parameters
            ----------
                variables : A list of either str, int, tuple, list or :class:`variable`
                default_bounds : Tuple of ints (lower bound, upper bound)

            Notes
            -----
            - Every item in ``*variables`` that is not an instance of :class:`variable` will be converted to a string and used as an id.
            - List of variables are returned sorted.

            Examples
            --------
                >>> variable.from_mixed("a",4,("b","c"),variable("x",(1,2)))
                [variable(id="('b', 'c')", bounds=Bounds(lower=0, upper=1)), variable(id='4', bounds=Bounds(lower=0, upper=1)), variable(id='a', bounds=Bounds(lower=0, upper=1)), variable(id='x', bounds=Bounds(lower=1, upper=2))]

                >>> variable.from_mixed("a",4,variable("x",(2,4)), default_bounds=(-1, 1))
                [variable(id='4', bounds=Bounds(lower=-1, upper=1)), variable(id='a', bounds=Bounds(lower=-1, upper=1)), variable(id='x', bounds=Bounds(lower=2, upper=4))]

            Returns
            -------
                out : List[:class:`variable`]
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
    def from_json(data: typing.Dict[str, typing.Any], class_map) -> "variable":
        """
            Creates :class:`variable` from json format.
        """
        bounds = data.get('bounds', {'lower': 0, 'upper': 1})
        return variable(
            id=data['id'],
            bounds=(bounds['lower'], bounds['upper'])
        )


class SolutionVariable(variable):
    """
        The :class:`SolutionVariable` class is an extension to the :class:`variable` class where each variable can be assigned a value. 

        Methods
        -------
        to_json
        from_variable
    """
    value: int = None

    def __init__(self, id: str, bounds: typing.Tuple[int, int] = None, dtype: str = None, value: int = None):
        super().__init__(id, bounds, dtype)
        self.value = value

    def __eq__(self, other):
        return self.id == other.id and self.value == other.value

    def to_json(self) -> typing.Dict[str, typing.Any]:
        d = super().to_json()
        d['value'] = self.value
        return d

    @staticmethod
    def from_variable(variable: variable, value: int) -> "SolutionVariable":
        return SolutionVariable(variable.id, variable.bounds, value=value)