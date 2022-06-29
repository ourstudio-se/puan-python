import ast
import string
import math
import hashlib
import base64
import pickle
import gzip
import itertools
import more_itertools
import typing
import operator
import puan
import puan.ndarray
import functools
import maz
import numpy
from dataclasses import dataclass, field
import puan.logicfunc as logicfunc

numpy.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))

default_min_int: int = numpy.iinfo(numpy.int16).min
default_max_int: int = numpy.iinfo(numpy.int16).max

class _Constraint(tuple):

    """
        Let variables=["x","y","z"] be variable list and then constraint ((0,1,2),(1,1,-2),0) 
        represents the linear inequality x+y-2z>=0
    """

    def __new__(cls, dtypes, index, values, b, id) -> dict:
        return tuple.__new__(cls, (tuple(dtypes,), tuple(index), tuple(values), b, id))

    def __lt__(self, other):
        return self.id < other.id

    @property
    def b(self):
        return self[3]

    @property
    def index(self):
        return self[1]

    @property
    def values(self):
        return self[2]

    @property
    def id(self):
        return self[4]

    @property
    def dtypes(self):
        return self[0]

    def __repr__(self):
        # return f"{self.id} = {' '.join((map(''.join, (zip(map(str, self.values), map(lambda x: f'({x})', self.index), map(lambda x: f'({x})', self.dtypes))))))} >= {self.b}"
        return f"{self.__class__.__name__}(dtypes={self.dtypes}, index={self.index}, values={self.values}, b={self.b}, id={self.id})"

    def extend(self):
        if self.id in self.index:
            return self
        else:
            if any(map(lambda x: x == 1, self.dtypes)):
                return _Constraint(
                    self.dtypes + (0,),
                    self.index + (self.id,), 
                    self.values + [(default_min_int-self.b,), (default_min_int,)][self.b > 0], 
                    [default_min_int, default_min_int+self.b][self.b > 0], 
                    self.id
                )
            else:
                return _Constraint(
                    self.dtypes + (0,),
                    self.index + (self.id,), 
                    self.values + (-self.b if self.b > 0 else -len(self.values),), 
                    0 if self.b > 0 else -len(self.values)+self.b, 
                    self.id
                )

class _CompoundConstraint(tuple):

    """
        The compound constraint (1, 1, [((0,1),(-1,-1),-1),((2,3,4),(1,1,1),3)])
        represents the union of linear inequalities: 
            [
                [-1,-1,-1, 0, 0, 0],
                [ 3, 0, 0, 1, 1, 1],
            ]

        The union is told by the first index in the outer most tuple (1), which really means
        first constraint ((0,1),(-1,-1),-1) + second constraint ((2,3,4),(1,1,1),3) >= 1. 
    """

    def __new__(cls, b: int, sign: int, constraints: typing.List[_Constraint], id: int) -> dict:
        return tuple.__new__(cls, (b, sign, list(itertools.starmap(_Constraint, constraints)), id))

    def __repr__(self) -> str:
        constraints_repr = ",".join(map(maz.compose(functools.partial(operator.add, "\n\t\t"), _Constraint.__repr__), self.constraints))
        return f"_CompoundConstraint(\n\tb={self.b},\n\tsign={self.sign},\n\tconstraints=[{constraints_repr}\n\t],\n\tid={self.id}\n)"

    @property
    def id(self):
        return self[3]

    @property
    def b(self):
        return self[0]

    @property
    def sign(self):
        return self[1]

    @property
    def constraints(self):
        return self[2]

    def transform(self):
        return _CompoundConstraint(len(self.constraints) + 1, 1,
            itertools.chain(
                [
                    _Constraint(
                        itertools.repeat(0, len(self.constraints)),
                        map(operator.attrgetter("id"), self.constraints),
                        itertools.repeat(self.sign, len(self.constraints)),
                        self.b,
                        self.id
                    )
                ],
                map(operator.methodcaller("extend"), self.constraints)
            ),
            self.id
        )

    @staticmethod
    def compose(*constraints, b: int, sign: int, id: int):
        return _CompoundConstraint(b, sign, map(maz.compose(operator.itemgetter(0), operator.attrgetter("constraints")), constraints), id)


@dataclass(frozen=True, repr=False)
class Proposition(puan.variable, list):

    """
        A Proposition is a variable resolving to true or false and is a collection of propositions joined by a sign (+/-) and a bias value
        such that it's underlying representation is e.g. A = a+b+c >= 3 (where all a,b and c are "atomic" propositions, sign is +, value is 3 and id is "A").
    """
    _id : str = None
    next_id = maz.compose(functools.partial(operator.add, "VAR"), str, itertools.count(1).__next__)

    def __init__(
        self, 
        *propositions: typing.List[typing.Union["Proposition", str]], 
        value: int = 1, 
        sign: int = 1, 
        id: str = None, 
        dtype: int = 0, 
        virtual: bool = None,
    ):
        # Allowing propositions to be either of type Proposition or type str
        # If str, then a new Proposition is instantiated with default dtype=bool and virtual=False
        object.__setattr__(self, "propositions", sorted(
                itertools.chain(
                    filter(lambda x: issubclass(x.__class__, Proposition), propositions),
                    map(
                        lambda x: Proposition(id=x, value=1),
                        filter(lambda x: isinstance(x, str), propositions)
                    )
                )
            )
        )
        
        # propositions as attributes on self
        list(map(lambda x: object.__setattr__(self, x.id, x), self.propositions))
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "sign", sign)
        super().__init__(id, dtype, virtual if virtual is not None else len(self.propositions) > 0)

    def __check__(self, v):
        if not isinstance(v, Proposition):
            raise TypeError(v)

    def __eq__(self, other):
        return self.id == getattr(other, "id", other)

    def __len__(self): 
        return len(self.propositions)

    def __getitem__(self, i): 
        return self.propositions[i]

    def __delitem__(self, i): 
        del self.propositions[i]
        self.value -= 1 if self.sign > 0 else -1

    def __setitem__(self, i, v):
        self.__check__(v)
        self.propositions[i] = v

    def __contains__(self,v):
        return v in self.propositions

    def append(self, i, v):
        self.__check__(v)
        self.value += 1 if self.sign > 0 else -1
        self.propositions.append(v)

    def insert(self, i, v):
        self.__check__(v)
        self.value += 1 if self.sign > 0 else -1
        self.propositions.insert(i, v)

    def pop(self, i):
        self.value -= 1 if self.sign > 0 else -1
        self.propositions.pop(i)

    @property
    def id(self) -> str:
        if self._id is None:
            return Proposition._id_generator(
                self.propositions, 
                self.value, 
                self.sign,
            )
        return self._id

    @id.setter
    def id(self, value):
        object.__setattr__(self, '_id', value)

    @property
    def atomic_propositions(self) -> filter:

        """
            We call a proposition "atomic" when it has no sub propositions
        """

        return filter(maz.compose(functools.partial(operator.eq, 0), len), self.propositions)

    @property
    def compound_propositions(self) -> filter:

        """
            We call a proposition "compound" when it has at least 1 sub proposition
        """

        return filter(maz.compose(functools.partial(operator.le, 1), len), self.propositions)

    def __repr__(self) -> str:
        eq="".join(map("".join, zip(itertools.repeat(["-","+"][self.sign > 0], len(self.propositions)), map(operator.attrgetter("id"), self.propositions))))
        return f"{self.__class__.__name__}(id='{self.id}', equation='{eq}>={self.value}')"

    def _id_generator(propositions, value, sign, n: int = 4, prefix: str = "VAR"):
        return prefix + hashlib.sha256(str("".join(map(operator.attrgetter("id"), sorted(propositions))) + str(value) + str(sign)).encode()).hexdigest()[:n]

    def _invert_sub(self) -> typing.Tuple[int, itertools.chain]:

        """
            Invert sub propositions. Compound proposition's
            get's inverted while atomic proposition's stay as they are.
        """

        return len(list(self.compound_propositions)), itertools.chain(
            map(
                operator.methodcaller("invert"),
                self.compound_propositions
            ),
            self.atomic_propositions
        )

    def _to_constraint(self, index_predicate) -> _Constraint:

        """
            Transforms this proposition into a _Constraint intstance.

            Examples
            --------
                >>> model = All("a","b","c","d",id="A")
                >>> model._to_constraint(model.variables_full().index)
                _Constraint(dtypes=(0, 0, 0, 0), index=(1, 2, 3, 4), values=(1, 1, 1, 1), b=4, id=0)
        """
        return _Constraint(
            map(
                maz.compose(
                    functools.partial(operator.mul, 1), 
                    functools.partial(operator.eq, 1),
                    operator.attrgetter("dtype"),
                ),
                self.propositions
            ),
            map(
                maz.compose(
                    index_predicate, 
                    operator.attrgetter("variable")
                ), 
                self.propositions
            ),
            itertools.repeat(self.sign, len(self.propositions)),
            self.value,
            index_predicate(self)
        )

    @property
    def size(self):

        """
            The size is the length of proposition list.

            Examples
            --------
                >>> model = All("x","y","z")
                >>> model.size
                3

            Returns
            -------
                out : int
        """

        return len(self.propositions)

    @property
    def variable(self) -> puan.variable:

        """
            This proposition as a puan.variable

            Returns
            -------
                out : puan.variable
        """
        return puan.variable(id=self.id, dtype=self.dtype, virtual=self.virtual)

    @property
    def variables(self):

        """
            The set of variables of this proposition.

            Examples
            --------
                >>> model = All("x","y","z",id="A")
                >>> model.variables
                [variable(id='A', dtype=0, virtual=True), variable(id='x', dtype=0, virtual=False), variable(id='y', dtype=0, virtual=False), variable(id='z', dtype=0, virtual=False)]

            Returns
            -------
                out : typing.Set[puan.variable]
        """

        return sorted(set(itertools.chain([self.variable], *map(operator.attrgetter("variables"), self.propositions))))

    @property
    def sub_variables(self) -> typing.List[puan.variable]:

        """
            The set of all variables of this sub propositions.

            Examples
            --------
                >>> model = All(Any("a","b", id="B"), All("y","z", id="C"), id="A")
                >>> model.sub_variables
                [variable(id='a', dtype=0, virtual=False), variable(id='b', dtype=0, virtual=False), variable(id='y', dtype=0, virtual=False), variable(id='z', dtype=0, virtual=False), variable(id='B', dtype=0, virtual=True), variable(id='C', dtype=0, virtual=True)]

            Returns
            -------
                out : typing.Set[puan.variable]
        """
        return sorted(set(itertools.chain(*map(operator.attrgetter("variables"), self.propositions))), key=lambda x: (x.virtual, x.id))
        
    def variables_full(self) -> typing.List[puan.variable]:

        """
            All, including sub proposition's, variables in this proposition model.

            Examples
            --------
                >>> All(All("a","b", id="B"), Any("c","d", id="C"), id="A").variables_full()
                [variable(id='A', dtype=0, virtual=True), variable(id='B', dtype=0, virtual=True), variable(id='C', dtype=0, virtual=True), variable(id='a', dtype=0, virtual=False), variable(id='b', dtype=0, virtual=False), variable(id='c', dtype=0, virtual=False), variable(id='d', dtype=0, virtual=False)]

            Notes
            -----
            This function assumes no variable reductions and returns the full variable space. 

            Returns
            -------
                out : typing.List[puan.variable]

        """

        return sorted(set(
            itertools.chain(
                [self.variable],
                *map(operator.attrgetter("variables"), self.propositions)
            )
        ))

    def invert(self) -> "Proposition":
        return self
 
    def to_compound_constraint(self, index_predicate: typing.Callable[[puan.variable], int], extend_top: bool = True) -> _CompoundConstraint:

        """
            Transforms into a compound linear inequality or constraint data format
            of _CompoundConstraint.

            Parameters
            ----------
            index_predicate : callable
                Function mapping variables of type puan.variable into indices.

            extend_top : bool
                If set to true, the top most proposition will be extended with its variable id

            Examples
            --------
                >>> model = All(All("a","b"), Any("c","d"))
                >>> sorted(model.to_compound_constraint(model.variables_full().index))
                [_Constraint(dtypes=(0, 0, 0), index=(3, 4, 0), values=(1, 1, -2), b=0, id=0), _Constraint(dtypes=(0, 0, 0), index=(0, 2, 1), values=(1, 1, -2), b=0, id=1), _Constraint(dtypes=(0, 0, 0), index=(5, 6, 2), values=(1, 1, -1), b=0, id=2)]

            Returns
            -------
                out : _CompoundConstraint
        """
        
        _constriant = self._to_constraint(index_predicate)
        return itertools.chain(
            [_constriant.extend() if extend_top else _constriant],
            *map(
                operator.methodcaller(
                    "to_compound_constraint", 
                    index_predicate=index_predicate,
                    extend_top=True, # Always extend children
                ), 
                self.compound_propositions
            )
        )
        
        # # C-func
        # transformed_constraint = _CompoundConstraint(
        #     *logicfunc.merge(
        #         _CompoundConstraint(self.value, self.sign, map(lambda x: x.constraints[0], sub_constraints), index_predicate(self)), 
        #         False
        #     )
        # )
        
    def to_polyhedron(self, active: bool = False, dtype = numpy.int32, support_variable_id: typing.Union[int, str] = 0) -> puan.ndarray.ge_polyhedron:

        """
            Transforms model into a polyhedron.

            Parameters
            ----------
            active : bool = False
                Assumes variable {self} set to true mening no merge with other system possible.

            dtype : numpy.dtypes = numpy.int32
                Data type of numpy matrix instance
            
            support_variable_id : typing.Union[int, str] = 0
                Variable id of the support (b) variable of the polyhedron

            Examples
            --------
                >>> ph = All(All("a","b",id="B"), Any("c","d",id="C"),id="A").to_polyhedron()
                >>> ph
                ge_polyhedron([[ 0, -2,  1,  1,  0,  0,  0,  0],
                               [ 0,  0, -2,  0,  1,  1,  0,  0],
                               [ 0,  0,  0, -1,  0,  0,  1,  1]])
                >>> ph.variables.tolist()
                [variable(id='0', dtype=1, virtual=True), variable(id='A', dtype=0, virtual=True), variable(id='B', dtype=0, virtual=True), variable(id='C', dtype=0, virtual=True), variable(id='a', dtype=0, virtual=False), variable(id='b', dtype=0, virtual=False), variable(id='c', dtype=0, virtual=False), variable(id='d', dtype=0, virtual=False)]
        """

        ph_vars = [puan.variable(str(support_variable_id),1,True)] + self.variables_full()
        compound_constraints = sorted(set(self.to_compound_constraint(ph_vars.index, extend_top=not active)))
        index = set(map(operator.attrgetter("id"),compound_constraints))
        M = numpy.zeros((len(compound_constraints), len(ph_vars)), dtype=dtype)
        for i,const in enumerate(compound_constraints):
            M[i, const.index + (0,)] = const.values + (const.b,)

        return puan.ndarray.ge_polyhedron(M, ph_vars, list(map(lambda x: ph_vars[x.id], compound_constraints)))

    def to_dict(self) -> typing.Dict[str, list]:

        """
            Transforms model into a dictionary representation.
            The key is the id of a proposition.
            The value is a list of three elements:
            1. sign of coeffs (e.g. sign=1 means a+b+c, sign=-1 means -a-b-c)
            2. sub propositions / variables (e.g. a,b,c)
            3. value of support vector (e.g. 3 as in a+b+c>=3)

            Examples
            --------
                >>> All(All("a","b", id="B"), Any("c","d", id="C"), id="A").to_dict()
                {'A': (1, ['B', 'C'], 2, 0, 1), 'B': (1, ['a', 'b'], 2, 0, 1), 'a': (1, [], 1, 0, 0), 'b': (1, [], 1, 0, 0), 'C': (1, ['c', 'd'], 1, 0, 1), 'c': (1, [], 1, 0, 0), 'd': (1, [], 1, 0, 0)}

            Returns
            -------
                out : typing.Dict[str, list]
        """

        t = self.to_tuple()
        return {
            **{
                t[0]: t[1:],
            },
            **functools.reduce(lambda x,y: dict(x,**y), map(operator.methodcaller("to_dict"), self.propositions), {})
        }

    def to_tuple(self):

        """
            Tuple format consists of six elements:
            (0) id (string)
            (1) sign (1/-1)
            (2) sub propositions (strings)
            (3) support value
            (4) data type (0 - bool / 1 - integer)
            (5) is virtual (0/1)

            Examples
            --------
                >>> All("x","y","z",id="A").to_tuple()
                ('A', 1, ['x', 'y', 'z'], 3, 0, 1)
        """

        return (self.id, self.sign, list(map(operator.attrgetter("id"), self.propositions)), self.value, self.dtype, self.virtual * 1)

    def to_b64(self, str_decoding: str = 'utf8') -> str:

        """
            Packs data into a base64 string.

            Parameters
            ----------
                str_decoding: str = 'utf8'

            Returns
            -------
                out : str
        """
        return base64.b64encode(
            gzip.compress(
                pickle.dumps(
                    self,
                    protocol=pickle.HIGHEST_PROTOCOL,
                ),
                mtime=0,
            )
        ).decode(str_decoding)

    def assume(self, *fixed: typing.List[str]) -> "Proposition":

        """
            Assumes certian propositions inside model to be fixed at some value.

            Parameters
            ----------
                fixed : typing.List[str]
                    fixed is a list of id's representing the id of propositions that are fixed to True

            Examples
            --------
                >>> model = AtLeast("x","y","z", value=2, id="A")
                >>> model.assume("x")
                Proposition(id='A', equation='+y+z>=1')

                >>> model = All(Any("a","b",id="B"), Any("x","y",id="C"), id="A")
                >>> model.assume("a")
                Proposition(id='C', equation='+x+y>=1')

            Returns
            -------
                out : Proposition
        """
        polyhedron = self.to_polyhedron(True)
        assumed_vector = polyhedron.A.construct(
            *zip(fixed, (1,)*len(fixed))
        )
        assumed_polyhedron = polyhedron.reduce_columns(assumed_vector)
        return from_polyhedron(
            assumed_polyhedron.reduce(
                *assumed_polyhedron.reducable_rows_and_columns()
            ),
            id=self._id
        )

    def reduce(self, fixed: typing.Dict[puan.variable, int]) -> "Proposition":

        """
            Reduces proposition by recursively removing sub propositions given the fixed
            values in the `fixed` -dictionary.

            Parameters
            ----------
                fixed : typing.Dict[puan.variable, int]
                    A dictionary with a puan.variable as key and positive/negative as value. Value is indicating
                    that the variable has a fixed positive/negative value. No key given or value is zero then no fixed value.

            Examples
            --------
                >>> AtLeast("x","y","z", value=2, id="A").reduce({"x": 1})
                AtLeast(id='A', equation='+y+z>=1')

                >>> AtLeast("x","y","z", value=2, id="A").reduce({"x": -1})
                AtLeast(id='A', equation='+y+z>=2')

                >>> AtMost("x","y","z", value=2, id="A").reduce({"x": 1})
                AtMost(id='A', equation='-y-z>=-1')

                >>> AtMost("x","y","z", value=2, id="A").reduce({"x": -1})
                AtMost(id='A', equation='-y-z>=-2')

            Returns
            -------
                out : Proposition 
        """
        return self.__class__(
            *map(
                operator.methodcaller("reduce", fixed),
                filter(
                    maz.compose(
                        functools.partial(
                            operator.eq,
                            0
                        ),
                        maz.pospartial(
                            fixed.get,
                            [(1, 0)]
                        ),
                        operator.attrgetter("id")
                    ),
                    self.propositions
                )
            ),
            value=abs(self.value)-sum(
                map(
                    maz.compose(
                        functools.partial(
                            operator.le,
                            1
                        ),
                        maz.pospartial(
                            fixed.get,
                            [(1, 0)]
                        ),
                        operator.attrgetter("id"),
                    ),
                    self.propositions
                )
            ),
            id=self._id,
        )

    def flatten(self) -> typing.Iterable["Proposition"]:

        """
            Putting this proposition and all its sub propositions into a flat list of propositions.

            Examples
            --------
                >>> AtLeast(AtLeast("x","y",value=1,id="C"),AtLeast("a","b",value=1,id="B"),value=1,id="A").flatten()
                [AtLeast(id='A', equation='+B+C>=1'), AtLeast(id='B', equation='+a+b>=1'), AtLeast(id='C', equation='+x+y>=1'), Proposition(id='a', equation='>=1'), Proposition(id='b', equation='>=1'), Proposition(id='x', equation='>=1'), Proposition(id='y', equation='>=1')]

            Returns
            -------
                out : typing.Iterable[Proposition]
        """
        return sorted(
            set(
                itertools.chain(
                    [self],
                    *map(
                        operator.methodcaller("flatten"),
                        self.propositions
                    ),
                )
            )
        )

    def specify(self) -> typing.Union["AtLeast", "AtMost"]:

        """
            Returns a more specific variant of this proposition.

            Returns
            -------
                out : typing.Union[AtLeast, AtMost]
        """

        instance = [AtMost, AtLeast][self.sign >= 1](
            *self.propositions,
            value=abs(self.value),
            id=self._id
        )
        object.__setattr__(instance, "dtype", self.dtype)
        object.__setattr__(instance, "virtual", self.virtual)
        return instance

    @property
    def is_tautologi(self) -> bool:

        """
            Returns wheather or not this proposition is true, no matter the interpretation of its propositions.

            Notes
            -----
            Sub propositions are not taken into consideration.

            Examples
            --------
                >>> model = AtLeast("x","y",value=1)
                >>> model.is_tautologi
                False

                >>> model = AtMost("x","y",value=1)
                >>> model.is_tautologi
                False

                >>> model = AtMost("x","y","z",value=3)
                >>> model.is_tautologi
                True

                >>> model = AtLeast("x",value=0)
                >>> model.is_tautologi
                True

                >>> model = AtMost("x","y",value=2)
                >>> model.is_tautologi
                True

            Returns
            -------
                out : bool
        """
        return ((self.sign > 0) & (self.value <= 0)) | ((self.sign < 0) & (-self.value >= len(self.propositions)))

    @property
    def is_contradiction(self) -> bool:

        """
            Returns wheather or not this proposition is false, no matter the interpretation of its propositions.

            Notes
            -----
            Sub propositions are not taken into consideration.

            Examples
            --------
                >>> model = AtLeast("x","y",value=1)
                >>> model.is_contradiction
                False

                >>> model = AtMost("x","y",value=1)
                >>> model.is_contradiction
                False

                >>> model = AtLeast("x","y",value=3)
                >>> model.is_contradiction
                True

                >>> model = AtMost("x","y",value=-1)
                >>> model.is_contradiction
                True

            Returns
            -------
                out : bool
        """
        return ((self.sign > 0) & (self.value > len(self.propositions))) | ((self.sign < 0) & (self.value > 0))

    def to_short(self) -> tuple:

        """
            `short` is a tuple format with six element types:
            (0) id, (1) sign, (2) propositions, (3) support value, (4) dtype and (5) is-virtual.

            Notes
            -----
                to_short does not include sub propositions if any

            Examples
            --------
                >>> All("x","y","z",id="A").to_short()
                ('A', 1, ['x', 'y', 'z'], 3, 0, 1)

            Returns
            -------
                out : tuple
        """
        return (self.id, self.sign, list(map(operator.attrgetter("id"), self.propositions)), self.value, self.dtype, self.virtual * 1)


    def to_text(self) -> str:

        """
            Returns a readable, concise, version controllable text format.

            Returns
            -------
                out : str
        """
        return "\n".join(
            map(
                maz.compose(
                    str,
                    operator.methodcaller("to_short")
                ),
                set(self.flatten())
            )
        )

class AtLeast(Proposition):

    """
        AtLeast is a Proposition which takes propositions and represents a lower bound on the 
        result of those propositions. For example, select at least one of x, y and z would be defined
        as AtLeast("x","y","z", value=1) and represented as x+y+z >= 1.
    """

    def __init__(self, *propositions: typing.List[typing.Union["Proposition", puan.variable]], value: int, id: str = None):
        super().__init__(*propositions, value=value, sign=1, id=id)
    
    def invert(self) -> "AtMost":
        
        """
            Inverts (or negates) proposition.

            Examples
            --------
                >>> AtLeast(*list("xyz"), value=2, id="A").invert()
                AtMost(id='A', equation='-x-y-z>=-1')

            Returns
            -------
                out : Proposition
        """

        if any(self.compound_propositions):
            return AtLeast(*map(operator.methodcaller("invert"), self.compound_propositions), value=-(self.value-1)+len(self.propositions), id=self._id)
        else:    
            return AtMost(*self.propositions, value=(self.value-1), id=self._id)

class AtMost(Proposition):

    """
        AtMost is a Proposition which takes propositions and represents a lower bound on the 
        result of those propositions. For example, select at least one of x, y and z would be defined
        as AtMost("x","y","z", value=2) and represented as -x-y-z >= -2.
    """

    def __init__(self, *propositions: typing.List[typing.Union["Proposition", puan.variable]], value: int, id: str = None):
        super().__init__(*propositions, value=-value, sign=-1, id=id)

    def invert(self) -> AtLeast:
        return AtLeast(
            *itertools.chain(map(operator.methodcaller("invert"), self.compound_propositions), self.propositions), 
            value=abs(self.value)+1,
            id=self._id,
        )

class All():

    """
        'All' is a Proposition representing a conjunction of all given propositions.
        'All' is represented by an AtLeast -proposition with value set to the number of given propositions.
        For example, x = All("x","y","z") is the same as y = AtLeast("x","y","z",value=3) 
    """

    def __new__(cls, *propositions, id: str = None):
        return AtLeast(*set(propositions), value=len(set(propositions)), id=id)

class Any():

    """
        'Any' is a Proposition representing a disjunction of all given propositions.
        'Any' is represented by an AtLeast -proposition with value set to 1.
        For example, Any("x","y","z") is the same as AtLeast("x","y","z",value=1) 
    """

    def __new__(cls, *propositions, id: str = None):
        return AtLeast(*propositions, value=1, id=id)

class Imply():

    """
        Imply is the implication logic operand and has two main inputs: condition and consequence.
        For example, if x is selected then y and z must be selected could be defined with the Imply -class
        as Imply("x", All("y","z")). 
    """

    def __new__(cls, condition: Proposition, consequence: Proposition, id: str = None):
        return Any(
            (condition if isinstance(condition, Proposition) else All(condition)).invert(), 
            consequence if isinstance(consequence, Proposition) else All(consequence), 
            id=id
        )

    @staticmethod
    def from_cicJE(data: dict, id_ident: str = "id") -> "Imply":

        """
            This function converts a cicJE into a Imply-data format.

            Parameters
            ----------
            data : dict

            id_ident : str = "id"
                The id identifier inside a component

            Examples
            --------
                >>> Imply.from_cicJE({
                ...     "id": "someId",
                ...     "condition": {
                ...         "relation": "ALL",
                ...         "subConditions": [
                ...             {
                ...                 "relation": "ALL",
                ...                 "components": [
                ...                     {"id": "a"},
                ...                     {"id": "b"},
                ...                 ]
                ...             }
                ...         ]
                ...     },
                ...     "consequence": {
                ...         "ruleType": "REQUIRES_ALL",
                ...         "components": [
                ...             {"id": "x"},
                ...             {"id": "y"},
                ...             {"id": "z"},
                ...         ]
                ...     }
                ... })
                AtLeast(id='someId', equation='+VAR1805+VARb6a0>=1')

            Returns
            -------
                out : Imply
            
        """
        rule_type_map = {
            "REQUIRES_ALL": lambda x,id: All(*x,id=id),
            "REQUIRES_ANY": lambda x,id: Any(*x,id=id),
            "ONE_OR_NONE": lambda x,id: AtMost(*x,value=1,id=id),
            "FORBIDS_ALL":  lambda x,id: Any(*x,id=id).invert(),
            "REQUIRES_EXCLUSIVELY": lambda x,id: Xor(*x,id=id)
        }
        cmp2prop = lambda x: Proposition(id=x[id_ident], dtype=[0,1]["dtype" in x and x['type'] == "int"], virtual=[False,True]["virtual" in x and x['virtual']])
        relation_fn = lambda x: [Any, All][x.get("relation", "ALL") == "ALL"]
        consequence = rule_type_map[data.get('consequence', {}).get("ruleType")](
            map(cmp2prop, data.get('consequence',{}).get('components')),
            data.get("consequence", {}).get("id", None)
        )
        if "condition" in data:
            condition_outer_cls = relation_fn(data['condition'])
            condition_inner_cls = list(map(
                lambda x: relation_fn(x)(*map(cmp2prop, x.get('components', [])), id=x.get("id", None)),
                data['condition'].get("subConditions", [])
            ))
            if len(condition_inner_cls) > 0:
                return Imply(
                    condition=condition_outer_cls(*condition_inner_cls, id=data.get('condition', {}).get("id", None)) if len(condition_inner_cls) > 1 else condition_inner_cls[0],
                    consequence=consequence,
                    id=data.get("id", None)
                )
            else:
                return consequence
        else:
            return consequence

class Xor():

    """
        Xor is restricting all propositions within to be selected exactly once.
        For example, Xor(x,y,z) means that x, y and z must be selected exactly once.
    """

    def __new__(cls, *propositions, id: str = None):
        return All(
            AtLeast(*propositions, value=1, id=f"{id}_LB" if id is not None else None), 
            AtMost(*propositions, value=1, id=f"{id}_UB" if id is not None else None),
            id=id
        )

class XNor():

    """
        XNor is a negated Xor. In the special case of two propositions, this is equivalent as a biconditional logical connective (<->).
        For example, XNor(x,y) means that only none or both are true.
    """

    def __new__(cls, *propositions, id: str = None):
        return Xor(*propositions).invert(id=id)

class Not():

    """
        Not is restricting propositions to never be selected.
        For example, Not("x","y","z") means that x, y or z can never be selected.
        Note that Not(x) is not necessarily equivilent to x.invert() (but could be).
    """

    def __new__(cls, *propositions, id: str = None):
        return AtMost(*propositions, value=0, id=id)


def from_b64(base64_str: str) -> typing.Any:

    """
        Unpacks base64 string `base64_str` into some data.

        Parameters
        ----------
            base64_str: str

        Returns
        -------
            out : dict
    """
    return pickle.loads(
        gzip.decompress(
            base64.b64decode(
                base64_str.encode()
            )
        )
    )

def delinearize(variables: list, row: numpy.ndarray, index: puan.variable) -> tuple:
    """
        When linearizing expression A: x+y+z>=3 one'll get -3A+x+y+z>=0. In other words,
        we'll rewrite the mixed linear logic expression A -> (x+y+z>=3) to a linear expression.
        When delinearizing we do it backwards with the assumptions:
            - linearized expression came from either an AtMost or an AtLeast
            - linearized expression constants are either -1 or 1

        Notes
        -----
        Support vector index assumed to be 0
        
        Examples
        --------
        When delinearizing -3a+x+y+z>=0
            >>> delinearize(puan.variable.from_strings(*list("0axyz")), numpy.array([0,-3,1,1,1]), puan.variable("a",0,True)) 
            (1, ['x', 'y', 'z'], 3, 0, 1)
        
        When delinearizing +x+y+z>=1. *Notice no change*.
            >>> delinearize(puan.variable.from_strings(*list("0axyz")), numpy.array([1,0,1,1,1]), puan.variable("a",0,True))
            (1, ['x', 'y', 'z'], 1, 0, 1)

        Returns
        -------
            out : tuple
    """
    if index in variables:
        row[0] -= row[variables.index(index)]
        row[variables.index(index)] = 0

    b,a = row[0], row[1:]
    sign = [-1,1][b > 0]

    return (sign, list(map(lambda i: variables[1:][i].id, numpy.argwhere(a == sign).T[0])), b, index.dtype, index.virtual * 1)

def from_polyhedron(polyhedron: puan.ndarray.ge_polyhedron, id: str = None) -> Proposition:

    """
        Transforms a polyhedron to a compound propositions.

        Notes
        -----
            Requires no row compressions in polyhedron.

        Returns
        -------
            out : Proposition
    """

    return from_dict(
        dict(
            zip(
                map(
                    operator.attrgetter("id"), 
                    polyhedron.index
                ),
                itertools.starmap(
                    functools.partial(
                        delinearize,
                        polyhedron.variables.tolist(),
                    ), 
                    zip(polyhedron, polyhedron.index)
                )
            )
        ),
        id=id
    )

def from_tuple(data: typing.Union[list, tuple]) -> Proposition:

    """
        A tuple(/list) data type of length 3 including (sign, variables(list), b-value)
        and has a 1-1 mapping to a compound proposition object.

        See also
        --------
            Proposition.to_tuple
            from_text
            from_b64
            from_dict
            from_polyhedron

        Examples
        --------
            >>> from_tuple(('A', 1, ['b','c'], 1, 0, 0))
            Proposition(id='A', equation='+b+c>=1')

        Returns
        -------
            out : Proposition
    """
    return Proposition(
        *data[2], 
        value=abs(data[3]), 
        id=data[0], 
        sign=data[1], 
        dtype=data[4],
        virtual=data[5]
    )

def from_text(text: str, id: str = None) -> Proposition:

    """
        Returns a proposition from text format

        Returns
        -------
            out : Proposition
    """
    parsed = list(
        map(
            ast.literal_eval,
            text.split("\n")
        )
    )
    return from_dict(
        dict(
            zip(
                map(
                    operator.itemgetter(0),
                    parsed
                ),
                map(
                    operator.itemgetter(
                        slice(1,None),
                    ),
                    parsed
                )
            )
        )
    )

def from_dict(d: dict, id: str = None) -> Proposition:

    """
        Transform from dictionary to a compound proposition.
        Values data type is following:
        [int, [str...], int] where 0 is the sign value,
        1 contains the variable names and 2 is the support vector value.

        Examples
        --------
            >>> from_dict({'a': [1, ['b','c'], 1, 0, 0], 'b': [1, ['x','y'], 1, 0, 0], 'c': [1, ['p','q'], 1, 0, 0]})
            Proposition(id='a', equation='+b+c>=1')

        Returns
        -------
            out : Proposition
    """
    d_conv = dict(
        zip(
            d.keys(),
            map(
                from_tuple,
                itertools.starmap(
                    maz.compose(
                        tuple,
                        more_itertools.prepend
                    ), 
                    d.items()
                ),
            )
        )
    )
    composition = All(
        *filter(
            lambda v: not any(map(lambda x: v.id in x, d_conv.values())),
            map(
                lambda v: v.__class__(
                    *map(
                        lambda x: d_conv.get(x.id, x),
                        v.propositions
                    ),
                    value=abs(v.value),
                    id=v.id,
                ),
                d_conv.values()
            )
        ),
        id=id
    )
    if len(composition) == 1:
        return composition[0]

    return composition