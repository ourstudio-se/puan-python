import hashlib
import base64
import pickle
import gzip
import itertools
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
                    0 if self.b > 0 else -len(self.values), 
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

@dataclass(frozen=True)
class Proposition(puan.variable):

    """
        A Proposition is an abstract class and an extension of the puan.variable class with the intention
        of being the atomic version of a compound proposition. When evaluated, the Proposition
        will take on a value and is later at some point evaluated to either true or false.
    """

    dtype: typing.Union[bool, int] = bool
    virtual: bool = False

    @property
    def variable(self) -> puan.variable:

        """
            Variable representation of this Proposition

            Returns
            -------
                out : puan.variable
        """

        return puan.variable(self.id, self.dtype, self.virtual)

    @property
    def variables(self) -> typing.List[puan.variable]:

        """
            Variable's representation of this Proposition, 
            including supporting variables.

            Returns
            -------
                out : typing.List[puan.variable]
        """
        return [self.variable]

    # def to_compound_constraint(self, index_predicate: typing.Callable[[puan.variable], int]) -> _CompoundConstraint:

    #     """
    #         Transforms into a compound linear inequality or constraint data format
    #         of _CompoundConstraint.
    #     """

    #     return _CompoundConstraint(1,1, [
    #         _Constraint(
    #             ([0,1][self.dtype == int],),
    #             (index_predicate(self.variable),),
    #             (1,), 
    #             1 if self.dtype == bool else default_min_int, 
    #             index_predicate(self.variable)
    #         )
    #     ], index_predicate(self.variable))

    def to_dict(self):
        return {}

@dataclass(frozen=True, repr=False)
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

    def __repr__(self) -> str:
        eq="".join(map("".join, zip(itertools.repeat(["-","+"][self.sign > 0], len(self.propositions)), map(operator.attrgetter("id"), self.propositions))))
        return f"{self.__class__.__name__}(id='{self.id}', equation='{eq}>={self.value}')"

    def _invert_sub(self) -> itertools.chain:

        """
            Invert sub propositions. CompoundProposition's
            get's inverted while Proposition's stay as they are.
        """

        return itertools.chain(
            map(
                operator.methodcaller("invert"),
                filter(
                    lambda x: issubclass(x.__class__, CompoundProposition),
                    self.propositions,
                )
            ),
            filter(
                maz.pospartial(isinstance, [(1, Proposition)]), 
                self.propositions
            )
        )

    def _to_constraint(self, index_predicate) -> _Constraint:

        """
            Transforms this proposition into a _Constraint intstance.
        """
        return _Constraint(
            map(
                maz.compose(
                    functools.partial(operator.mul, 1), 
                    functools.partial(operator.eq, int),
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
    def variables(self):

        """
            The set of variables of this proposition.

            Examples
            --------
                >>> model = All("x","y","z")
                >>> model.variables
                {variable(id='VAR8', dtype=<class 'bool'>, virtual=True), variable(id='y', dtype=<class 'bool'>, virtual=False), variable(id='z', dtype=<class 'bool'>, virtual=False), variable(id='x', dtype=<class 'bool'>, virtual=False)}

            Returns
            -------
                out : typing.Set[puan.variable]
        """

        return set(itertools.chain([self.variable], *map(operator.attrgetter("variables"), self.propositions)))

    @property
    def sub_variables(self) -> typing.List[puan.variable]:

        """
            The set of all variables of this sub propositions.

            Examples
            --------
                >>> model = All(Any("a","b"),"y","z")
                >>> model.sub_variables
                {variable(id='a', dtype=<class 'bool'>, virtual=False), variable(id='b', dtype=<class 'bool'>, virtual=False)}

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
                >>> All(All("a","b"), Any("c","d")).variables_full()
                [variable(id='VAR1', dtype=<class 'bool'>, virtual=True), variable(id='VAR2', dtype=<class 'bool'>, virtual=True), variable(id='VAR3', dtype=<class 'bool'>, virtual=True), variable(id='a', dtype=<class 'bool'>, virtual=False), variable(id='b', dtype=<class 'bool'>, virtual=False), variable(id='c', dtype=<class 'bool'>, virtual=False), variable(id='d', dtype=<class 'bool'>, virtual=False)]

            Notes
            -----
            This function assumes no variable reductions and returnes the full variable space. 

            Returns
            -------
                out : typing.List[puan.variable]

        """

        return sorted(set(
            itertools.chain(
                [self.variable],
                *map(operator.attrgetter("variables"), self.propositions)
            )
        ), key=lambda x: (1-x.virtual, x.id))

    def invert(self) -> "CompoundProposition":

        """
            Inverts (or negates) proposition.

            Examples
            --------
                >>> AtLeast(*list("xyz"), value=2, id="A").invert()
                AtMost(id='A', equation='-x-y-z>=-1')

            Returns
            -------
                out : CompoundProposition
        """

        raise NotImplementedError()
 
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
                >>> model.to_compound_constraint(model.variables_all().index)
                _CompoundConstraint(
                    b=3,
                    sign=1,
                    constraints=[
                        _Constraint(dtypes=(0, 0), index=(0, 1), values=(1, 1), id=2),
                        _Constraint(dtypes=(0, 0), index=(3, 4, 0), values=(1, 1, -2), id=0),
                        _Constraint(dtypes=(0, 0), index=(5, 6, 1), values=(1, 1, -1), id=1)
                    ],
                    id=2
                )

            Returns
            -------
                out : CompoundProposition
        """

        # if all(map(lambda x: not hasattr(x, "propositions"), self.propositions)):
        #     return self._to_constraint(index_predicate)

        _constriant = self._to_constraint(index_predicate)
        return itertools.chain(
            [_constriant.extend() if extend_top else _constriant],
            *map(
                operator.methodcaller(
                    "to_compound_constraint", 
                    index_predicate=index_predicate,
                    extend_top=True, # Always extend children
                ), 
                filter(
                    lambda x: isinstance(x, CompoundProposition), 
                    self.propositions,
                )
            )
        )
        
        # # C-func
        # transformed_constraint = _CompoundConstraint(
        #     *logicfunc.merge(
        #         _CompoundConstraint(self.value, self.sign, map(lambda x: x.constraints[0], sub_constraints), index_predicate(self)), 
        #         False
        #     )
        # )
        
        # Python-func
        # transformed_constraint = _CompoundConstraint(self.value, self.sign, map(lambda x: x.constraints[0], sub_constraints), index_predicate(self)).transform()
        
        
        # return _CompoundConstraint(
        #     transformed_constraint.b, 
        #     transformed_constraint.sign, 
        #     itertools.chain(
        #         transformed_constraint.constraints,
        #         itertools.chain(*map(lambda x: x.constraints[1:], sub_constraints)), 
        #     ),
        #     transformed_constraint.id
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
                >>> ph = All(All("a","b"), Any("c","d")).to_polyhedron()
                >>> ph
                ge_polyhedron([[ 2,  1,  1,  0,  0,  0,  0,  0],
                               [ 0,  0, -1,  0,  0,  0,  1,  1],
                               [ 0, -2,  0,  0,  1,  1,  0,  0]], dtype=int32)
                >>> ph.variables
                [variable(id=0, dtype=<class 'int'>, virtual=True), variable(id='VAR1', dtype=<class 'bool'>, virtual=True), variable(id='VAR2', dtype=<class 'bool'>, virtual=True), variable(id='VAR3', dtype=<class 'bool'>, virtual=True), variable(id='a', dtype=<class 'bool'>, virtual=False), variable(id='b', dtype=<class 'bool'>, virtual=False), variable(id='c', dtype=<class 'bool'>, virtual=False), variable(id='d', dtype=<class 'bool'>, virtual=False)]
        """

        ph_vars = [puan.variable(str(support_variable_id),int,True)] + self.variables_full()
        compound_constraints = set(self.to_compound_constraint(ph_vars.index, extend_top=not active))
        index = set(map(operator.attrgetter("id"),compound_constraints))
        M = numpy.zeros((len(compound_constraints), len(ph_vars)), dtype=dtype)
        for i,const in enumerate(compound_constraints):
            M[i, const.index + (0,)] = const.values + (const.b,)

        return puan.ndarray.ge_polyhedron(M, ph_vars, list(map(lambda x: ph_vars[x.id], compound_constraints)))

    def to_dict(self):

        """
            Transforms model into a dictionary representation.

            Examples
            --------
                >>> All(All("a","b"), Any("c","d")).to_dict()
                {'VAR7': 'VAR6+VAR5>=2', 'VAR6': 'c+d>=1', 'VAR5': 'a+b>=2'}
        """

        return {
            **{
                self.variable.id: f"{['-',''][self.value > 0]}{(['-','+'][self.value > 0]).join(map(lambda x: x.variable.id, self.propositions))}>={self.value}"
            },
            **functools.reduce(lambda x,y: dict(x,**y), map(operator.methodcaller("to_dict"), self.propositions))
        }

    def pack_b64(self, str_decoding: str = 'utf8') -> str:

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

    @staticmethod
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


class AtLeast(CompoundProposition):

    """
        AtLeast is a CompoundProposition which takes propositions and represents a lower bound on the 
        result of those propositions. For example, select at least one of x, y and z would be defined
        as AtLeast("x","y","z", value=1) and represented as x+y+z >= 1.
    """

    def __init__(self, *propositions: typing.List[typing.Union["Proposition", puan.variable]], value: int, id: str = None):
        super().__init__(*propositions, value=value, sign=1, id=id)
    
    def invert(self) -> "AtMost":
        return AtMost(*self._invert_sub(), value=(self.value-1), id=self.id)

class AtMost(CompoundProposition):

    """
        AtMost is a CompoundProposition which takes propositions and represents a lower bound on the 
        result of those propositions. For example, select at least one of x, y and z would be defined
        as AtMost("x","y","z", value=2) and represented as -x-y-z >= -2.
    """

    def __init__(self, *propositions: typing.List[typing.Union["Proposition", puan.variable]], value: int, id: str = None):
        super().__init__(*propositions, value=-value, sign=-1, id=id)

    def invert(self) -> AtLeast:
        return AtLeast(*self._invert_sub(), value=abs(self.value)+1, id=self.id)

class All():

    """
        'All' is a CompoundProposition representing a conjunction of all given propositions.
        'All' is represented by an AtLeast -proposition with value set to the number of given propositions.
        For example, x = All("x","y","z") is the same as y = AtLeast("x","y","z",value=3) 
    """

    def __new__(cls, *propositions, id: str = None):
        return AtLeast(*set(propositions), value=len(set(propositions)), id=id)

class Any():

    """
        'Any' is a CompoundProposition representing a disjunction of all given propositions.
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

    def __new__(cls, condition: CompoundProposition, consequence: CompoundProposition, id: str = None):
        return Any(
            (condition if isinstance(condition, CompoundProposition) else All(condition)).invert(), 
            consequence if isinstance(consequence, CompoundProposition) else All(consequence), 
            id=id
        )

class Xor():

    """
        Xor is restricting all propositions within to be selected exactly once.
        For example, Xor("x","y","z") means that x, y and z must be selected exactly once.
    """

    def __new__(cls, *propositions, id: str = None):
        return All(
            AtLeast(*propositions, value=1), 
            AtMost(*propositions, value=1),
            id=id
        )

class Not():

    """
        Not is restricting propositions to never be selected.
        For example, Not("x","y","z") means that x, y or z can never be selected.
        Note that Not(x) is not necessarily equivilent to x.invert() (but could be).
    """

    def __new__(cls, *propositions, id: str = None):
        return AtMost(*propositions, value=0, id=id)
