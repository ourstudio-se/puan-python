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

    def assume(self, fixed: typing.Dict[str, int]) -> "Proposition":
        
        """
            Assumes something about variable's bounds and returns a new proposition with these new bounds set.
            Other variables, not declared in `new_variable_bounds`, may also get new bounds as a consequence from the ones set
            in `new_variable_bounds`.

            Parameters
            ----------
                new_variable_bounds : typing.Dict[str, Union[int, Tuple[int, int], puan.Bounds]]
                    A dict of ids and either ``int``, ``tuple`` or :class:`puan.Bounds` as bounds for the variable.

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

    def __eq__(self, obj):
        return self.as_tuple() == (obj.as_tuple() if issubclass(obj.__class__, Bounds) else obj)

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

    def __init__(
        self, 
        id: str, 
        bounds: typing.Optional[typing.Union[int, typing.Tuple[int, int], Bounds]] = None, 
        dtype: Dtype = None
    ):
        self.id = id
        if issubclass(bounds.__class__, Bounds):
            self.bounds = bounds
        elif issubclass(bounds.__class__, (int, numpy.integer)):
            self.bounds = Bounds(int(bounds), int(bounds))
        else:
            if dtype is not None:
                if bounds is not None:
                    if (dtype == "bool") != ((dtype == "bool") and (bounds == (0, 1))):
                        raise ValueError("Dtype is bool thus bounds must be (0, 1), got: {}".format(bounds))
                else:
                    bounds = {"int": default_int_bounds, "bool": (0, 1)}.get(dtype, (0, 1))
            elif bounds is None:
                bounds = (0, 1)
            
            if not issubclass(bounds.__class__, (tuple, numpy.ndarray, list)):
                raise ValueError(f"invalid data type for bounds, got `{bounds.__class__}`")

            self.bounds = Bounds(*bounds)

    def __hash__(self):
        return hash(self.id)+hash(self.bounds)

    def __lt__(self, other):
        return self.id < other.id

    def __eq__(self, other):
        return self.id == getattr(other, "id", other)

    def assume(self, fixed: typing.Dict[str, typing.Union[int, typing.Tuple[int, int], Bounds]]) -> Proposition:
        
        if self.id in fixed:
            return variable(
                id=self.id,
                bounds=fixed[self.id],
            )

        return self

    def evaluate(self, interpretation: typing.Dict[str, typing.Union[Bounds, typing.Tuple[int,int], int]]) -> Bounds:

        """
            Evaluates ``interpretation`` on this proposition. 

            Parameters
            ----------
                interpretation : Dict[str, Union[Bounds, Tuple[int,int], int]]
                    the values of the variables in the model to evaluate it for

            Examples
            --------
                >>> variable("a", bounds=(0,1)).evaluate({"x": 1})
                Bounds(lower=0, upper=1)

                >>> variable("a", bounds=(0,2)).evaluate({"a": 2})
                Bounds(lower=2, upper=2)

                >>> variable("a", bounds=(0,2)).evaluate({"a": (2,3)})
                Bounds(lower=2, upper=3)

                >>> variable("a", bounds=(0,2)).evaluate({"a": Bounds(1,1)})
                Bounds(lower=1, upper=1)

                >>> variable("a", bounds=(-1,0)).evaluate({"a": 1})
                Bounds(lower=1, upper=1)

                >>> variable("a", bounds=(1,1)).evaluate({"a": 0})
                Bounds(lower=0, upper=0)

                >>> variable("a", bounds=(1,1)).evaluate({"x": 1})
                Bounds(lower=1, upper=1)
            
            Returns
            -------
                out : Optional[int]
        """
        if self.id in interpretation:
            val = interpretation.get(self.id)
            if issubclass(val.__class__, (int, numpy.integer)):
                return Bounds(
                    interpretation.get(self.id), 
                    interpretation.get(self.id)
                )
            elif issubclass(val.__class__, tuple):
                return Bounds(*val)
            elif issubclass(val.__class__, Bounds):
                return val
            else:
                raise ValueError(f"extracted value from interpretation is not a valid type, got `{val.__class__}` type")
        return self.bounds

    def evaluate_propositions(
        self, 
        interpretation: typing.Dict[str, typing.Union[Bounds, typing.Tuple[int,int], int]],
        out: typing.Callable[[Bounds], typing.Union[Bounds, typing.Tuple[int,int], int]] = lambda x: x,
    ) -> typing.Dict[str, Bounds]:

        """
            Evaluates ``interpretation`` on this proposition. 

            Parameters
            ----------
                interpretation : Dict[Union[str, :class:`variable`], int]
                    the values of the variables in the model to evaluate it for
                out : Callback[[Bounds], Union[Bounds, Tuple[int,int], int]]
                    an optional callback function for changing output data type.

            Examples
            --------
                >>> variable("a", bounds=(0,1)).evaluate_propositions({"x": 1})
                {'a': Bounds(lower=0, upper=1)}

                >>> variable("a", bounds=(0,1)).evaluate_propositions({"a": (1,1)})
                {'a': Bounds(lower=1, upper=1)}

                >>> variable("a", bounds=(0,1)).evaluate_propositions({"a": Bounds(1,1)})
                {'a': Bounds(lower=1, upper=1)}

                >>> variable("a", bounds=(0,2)).evaluate_propositions({"a": 2})
                {'a': Bounds(lower=2, upper=2)}

                >>> variable("a", bounds=(-1,0)).evaluate_propositions({"a": 1})
                {'a': Bounds(lower=1, upper=1)}

                >>> variable("a", bounds=(1,1)).evaluate_propositions({"a": 0})
                {'a': Bounds(lower=0, upper=0)}

                >>> variable("a", bounds=(1,1)).evaluate_propositions({"x": 1})
                {'a': Bounds(lower=1, upper=1)}
            
            Returns
            -------
                out : Optional[int]
        """
        return {self.id: out(self.evaluate(interpretation))}

    def flatten(self) -> typing.List[Proposition]:

        """
            Returns all its propositions and their sub propositions as a unique list of propositions.

            Examples
            --------
                >>> variable("a", bounds=(0,2)).flatten()
                [variable(id='a', bounds=Bounds(lower=0, upper=2))]
        
            Returns
            -------
                out : List[Proposition]
        """

        return [self]

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