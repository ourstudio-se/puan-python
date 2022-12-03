import json
import numpy
import typing
import maz
import itertools
import dataclasses
from enum import Enum, IntEnum
from json import JSONEncoder

default_min_int     : int        = numpy.iinfo(numpy.int16).min
default_max_int     : int        = numpy.iinfo(numpy.int16).max
default_int_bounds  : (int, int) = (default_min_int, default_max_int)

def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)

_default.default = JSONEncoder().default
JSONEncoder.default = _default

class Dtype(str, Enum):
    """
        An :class:`Enum` class to define data type of variable, boolean and integer.  
    """
    BOOL = "bool"
    INT = "int"

class Proposition:

    # @property
    # def id(self) -> str:
    #     raise NotImplementedError()

    # @property
    # def bounds(self) -> typing.Tuple[int,int]:

    #     """
    #         Variable bounds of where this variable can obtain a value.
    #         A tuple of two integers decides the bounds where the lower bound
    #         is on index 0 and upper bound on index 1. Both lower and upper
    #         bound are inclusive.

    #         Returns
    #         -------
    #             out : Tuple[int, int]
    #     """

    #     raise NotImplementedError()

    def assume(self, fixed: typing.Dict[str, int]) -> "Proposition":
        
        """
            Fixes variables to a constant value and resolves consequences.

            Parameters
            ----------
                fixed : typing.Dict[str, int]

            Returns
            -------
                out : Proposition 
        """
        raise NotImplementedError()

    def to_short(self) -> typing.Tuple[str, int, object, int, typing.Tuple[int, int]]:
        
        """Short data type has (id, sign, propositions, value, bounds)"""
        raise NotImplementedError()

    def to_json(self) -> typing.Dict[str, typing.Any]:

        """JSON representation of class"""
        raise NotImplementedError()

    @staticmethod
    def from_json(data, class_map) -> typing.Any:

        """Convert from JSON into class object. Class map is a list of other classes maybe related to this class"""
        raise NotImplementedError()

@dataclasses.dataclass
class Bounds:
    """
        The :class:`Bounds` class defines lower and upper bounds of a :class:`variable`

        Raises
        ------
            ValueError
                If ``lower`` is greater than ``upper`` 

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

    def __hash__(self) -> int:
        return hash(self.lower)+hash(self.upper)

    def __iter__(self):
        return iter([self.lower, self.upper])

    @property
    def constant(self) -> typing.Optional[int]:

        """
            If lower and upper bounds are the same, that value is returned.
            Else None is returned.

            Examples
            --------
                >>> Bounds(0, 1).constant


                >>> Bounds(-2, -2).constant
                -2

            Returns
            -------
                out : Optional[int]
        """
        if self.lower == self.upper:
            return self.lower
        return None

    def as_tuple(self) -> typing.Tuple[int, int]:
        """
            Bounds as tuple

            Returns
            -------
                out : Tuple[int, int]
        """
        return (self.lower, self.upper)

@dataclasses.dataclass
class variable(Proposition):

    """
        The :class:`variable` class is a central key in all Puan packages. It consists of an id, :class:`Bounds` and data type (dtype).

        Raises
        ------
            ValueError
                If ``dtype`` is bool and bounds are not (0, 1)

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

    def __init__(self, id: str, bounds: typing.Tuple[int, int] = None, dtype: Dtype = None):
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
        return hash(self.id)+hash(self.bounds)

    def __lt__(self, other):
        return self.id < other.id

    def __eq__(self, other):
        return self.id == getattr(other, "id", other)

    def assume(self, fixed: typing.Dict[str, int]) -> Proposition:
        
        if self.id in fixed:
            return variable(
                id=self.id,
                bounds=Bounds(
                    lower=fixed[self.id],
                    upper=fixed[self.id],
                )
            )

        return self

    def to_json(self) -> typing.Dict[str, typing.Any]:

        """
            Converts into JSON data type.

            Examples
            --------
                >>> variable(id="x", bounds=Bounds(lower=0, upper=10)).to_json()
                {'id': 'x', 'bounds': {'lower': 0, 'upper': 10}}
        """

        d = dataclasses.asdict(self)
        if self.bounds.as_tuple() == (0,1):
            del d['bounds']
        return d

    def to_short(self) -> typing.Tuple[str, int, typing.List, int, typing.Tuple[int, int]]:
        """
            Converts into `short` data type.

            Examples
            --------
                >>> variable(id="x", bounds=Bounds(lower=0, upper=10)).to_short()
                ('x', 1, [], 0, (0, 10))
        """
        return (self.id, 1, [], 0, self.bounds.as_tuple())

    @staticmethod
    def support_vector_variable() -> "variable":
        """
            Default support vector variable settings.

            Examples
            --------
                >>> variable.support_vector_variable()
                variable(id=0, bounds=Bounds(lower=1, upper=1))

            Returns
            -------
                out : :class:`variable`
        """
        return variable(0, bounds=Bounds(lower=1, upper=1))

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
            - Every item `x` in ``*variables`` that is **not** an instance of :class:`variable` will be converted to a :class:`variable` with id set to `x` and default bounds.
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
    def from_json(data: typing.Dict[str, typing.Any], class_map: typing.List[typing.Type] = []) -> "variable":
        """
            Creates :class:`variable` from json format.

            Notes
            -----
                Bounds are defaulted to :class:`Bounds(lower=0, upper=1)` if not provided.
                ``class_map`` is ignored.

            Examples
            --------
                >>> variable.from_json({'id': 'x'})
                variable(id='x', bounds=Bounds(lower=0, upper=1))

                >>> variable.from_json({'id': 'x', 'bounds': {'lower': -2, 'upper': 3}})
                variable(id='x', bounds=Bounds(lower=-2, upper=3))
        """
        bounds = data.get('bounds', {'lower': 0, 'upper': 1})
        return variable(
            id=data['id'],
            bounds=(bounds['lower'], bounds['upper'])
        )
        

class Sign(IntEnum):
    """
        Sign, either positive or negative
    """
    POSITIVE = 1
    NEGATIVE = -1