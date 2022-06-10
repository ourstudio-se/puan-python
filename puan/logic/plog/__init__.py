import hashlib
import itertools
import typing
import operator
import puan
import puan.ndarray
import functools
import maz
import numpy
from dataclasses import dataclass, field

numpy.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))

default_min_int: int = numpy.iinfo(numpy.int16).min
default_max_int: int = numpy.iinfo(numpy.int16).max

class Constraint(tuple):

    """
        Let variables=["x","y","z"] be variable list and then constraint ((0,1,2),(1,1,-2),0) 
        represents the linear inequality x+y-2z>=0
    """

    def __new__(cls, dtypes, index, values, b, id) -> dict:
        return tuple.__new__(cls, (tuple(dtypes,), tuple(index), tuple(values), b, id))

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
        return f"{self.id} = {' '.join((map(''.join, (zip(map(str, self.values), map(lambda x: f'({x})', self.index), map(lambda x: f'({x.__name__})', self.dtypes))))))} >= {self.b}"

    def extend(self):
        if self.id in self.index:
            return self
        else:
            if any(map(lambda x: x == int, self.dtypes)):
                return Constraint(
                    self.dtypes + (bool,),
                    self.index + (self.id,), 
                    self.values + [(default_min_int-self.b,), (default_min_int,)][self.b > 0], 
                    [default_min_int, default_min_int+self.b][self.b > 0], 
                    self.id
                )
            else:
                return Constraint(
                    self.dtypes,
                    self.index + (self.id,), 
                    self.values + (-self.b if self.b > 0 else -len(self.values),), 
                    0 if self.b > 0 else -len(self.values), 
                    self.id
                )

class CompoundConstraint(tuple):

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

    def __new__(cls, b: int, sign: int, constraints: typing.List[Constraint], id: int) -> dict:
        return tuple.__new__(cls, (b, sign, list(constraints), id))

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
        return CompoundConstraint(len(self.constraints) + 1, 1,
            itertools.chain(
                [
                    Constraint(
                        itertools.repeat(bool, len(self.constraints)),
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
        return CompoundConstraint(b, sign, map(maz.compose(operator.itemgetter(0), operator.attrgetter("constraints")), constraints), id)

@dataclass(frozen=True)
class Proposition(puan.variable):

    """
        A Proposition is an abstract class and an extension of the puan.variable class with the intention
        of being the atomic version of a compound proposition. When evaluated, the Proposition
        will take on a value and is later at some point evaluated to either true or false.
    """

    id: str
    dtype: typing.Union[bool, int] = bool
    virtual: bool = False

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return self.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other) -> bool:
        return self.id == other.id

    @property
    def variable(self) -> puan.variable:

        """"""

        return puan.variable(self.id, self.dtype, self.virtual)

    @property
    def variables(self) -> typing.List[puan.variable]:
        return [self.variable]

    def invert(self):
        return self

    def to_compound_constraint(self, index_pred) -> CompoundConstraint:
        return CompoundConstraint(1,1, [
            Constraint((self.dtype,),(index_pred(self.variable),),(1,), 0 if self.dtype == bool else default_min_int, index_pred(self.variable))
        ], index_pred(self.variable))

    def to_dict(self):
        return {}

@dataclass(frozen=True)
class CompoundProposition(Proposition):

    """
        A Compound Proposition is a collection of propositions joined by a sign (+/-) and a bias value
        such that it's underlying representation is e.g. A = a+b+c >= 3 (where all a,b and c are propositions,
        sign is +, value is 3 and id is "A").
    """

    next_id = maz.compose(functools.partial(operator.add, "VAR"), str, itertools.count(1).__next__)

    def __init__(self, *propositions: typing.List[typing.Union[Proposition, str]], value: int, sign: int, id: str = None):
        super().__init__(CompoundProposition.next_id() if id is None else id, bool, True)
        
        # Allowing propositions to be either of type Proposition or type str
        # If str, then a new Proposition is instantiated with default dtype=bool and virtual=False
        self.propositions = list(
            itertools.chain(
                filter(lambda x: issubclass(x.__class__, Proposition), propositions),
                map(
                    lambda x: Proposition(x, bool, False),
                    filter(lambda x: isinstance(x, str), propositions)
                )
            )
        )
        self.value = value
        self.sign = sign

    # @property
    # def _id(self) -> str:
    #     return f"VAR{hashlib.shake_256((''.join(map(operator.attrgetter('id'), sorted(self.propositions)))+str(self.value)).encode()).hexdigest(2)}"

    @property
    def size(self):
        return len(self.propositions)

    @property
    def variables(self):
        return set(itertools.chain([self.variable], *map(operator.attrgetter("variables"), self.propositions)))

    def invert(self) -> "CompoundProposition":
        raise NotImplementedError()

    def to_compound_constraint(self, index_pred) -> CompoundConstraint:
        if all(map(lambda x: not hasattr(x, "propositions"), self.propositions)):
            return CompoundConstraint(1, 1, 
                [Constraint(
                    map(lambda x: x.dtype, self.propositions),
                    map(lambda x: index_pred(x.variable), self.propositions),
                    itertools.repeat(self.sign, len(self.propositions)),
                    self.value,
                    index_pred(self)
                )], 
                index_pred(self)
            )

        sub_constraints = list(map(operator.methodcaller("to_compound_constraint", index_pred=index_pred), self.propositions))
        transformed_constraint = CompoundConstraint(self.value, self.sign, map(lambda x: x.constraints[0], sub_constraints), index_pred(self)).transform()
        complete = CompoundConstraint(
            transformed_constraint.b, 
            transformed_constraint.sign, 
            itertools.chain(
                transformed_constraint.constraints,
                itertools.chain(*map(lambda x: x.constraints[1:], sub_constraints)), 
            ),
            transformed_constraint.id
        )
        return complete

    def to_polyhedron(self, dtype = numpy.int32, support_variable_id: typing.Union[int, str] = 0) -> puan.ndarray.ge_polyhedron:
        ph_vars = [puan.variable(support_variable_id,int,True)] + sorted(
            set(
                itertools.chain(
                    [self.variable],
                    *map(operator.attrgetter("variables"), self.propositions)
                )
            )
        )
        compound_constraint = self.to_compound_constraint(ph_vars.index)
        M = numpy.zeros((len(compound_constraint.constraints), len(ph_vars)), dtype=dtype)
        for i,const in enumerate(compound_constraint.constraints):
            M[i,const.index] = const.values
            M[i,0] = const.b

        return puan.ndarray.ge_polyhedron(M, ph_vars)

    def to_dict(self):
        return {
            **{
                self.variable.id: f"{['-',''][self.value > 0]}{(['-','+'][self.value > 0]).join(map(lambda x: x.variable.id, self.propositions))}>={self.value}"
            },
            **functools.reduce(lambda x,y: dict(x,**y), map(operator.methodcaller("to_dict"), self.propositions))
        }

class AtLeast(CompoundProposition):

    def __init__(self, *propositions: typing.List[typing.Union["Proposition", puan.variable]], value: int, id: str = None):
        super().__init__(*propositions, value=value, sign=1, id=id)

    def __str__(self) -> str:
        return f"{{{self.id}=({'+'.join(map(str, self.propositions))}>={self.value})}}"
    
    def invert(self) -> "AtMost":
        return AtMost(*map(operator.methodcaller("invert"), self.propositions), value=(self.value-1), id=self.id)

class AtMost(CompoundProposition):

    def __init__(self, *propositions: typing.List[typing.Union["Proposition", puan.variable]], value: int, id: str = None):
        super().__init__(*propositions, value=-value, sign=-1, id=id)

    def __str__(self) -> str:
        return f"{{{self.id}=(-{'-'.join(map(str, self.propositions))}>={self.value})}}"

    def invert(self) -> AtLeast:
        return AtLeast(*map(operator.methodcaller("invert"), self.propositions), value=abs(self.value)+1, id=self.id)

class All():

    def __new__(cls, *propositions, id: str = None):
        return AtLeast(*set(propositions), value=len(set(propositions)), id=id)

class Any():

    def __new__(cls, *propositions, id: str = None):
        return AtLeast(*propositions, value=1, id=id)

class Imply():

    def __new__(cls, cond: CompoundProposition, cons: CompoundProposition, id: str = None):
        return Any(cond.invert(), cons, id=id)

class Xor():

    def __new__(cls, *propositions, id: str = None):
        return All(
            AtLeast(*propositions, value=1), 
            AtMost(*propositions, value=1),
            id=id
        )

class Not():

    def __new__(cls, *propositions, id: str = None):
        return AtMost(*propositions, value=0, id=id)
