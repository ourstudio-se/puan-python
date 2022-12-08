import enum
import hashlib
import maz
import base64
import gzip
import graphlib
import pickle
import functools
import itertools
import typing
import operator
import numpy as np
import puan
import puan.ndarray as pnd
import puan_rspy as pr
import more_itertools
from dataclasses import dataclass
from collections import Counter

class PropositionValidationError(str, enum.Enum):

    CIRCULAR_DEPENDENCIES = "CIRCULAR_DEPENDENCIES"
    AMBIVALENT_VARIABLE_DEFINITIONS = "AMBIVALENT_VARIABLE_DEFINITIONS"
    NON_UNIQUE_SUB_PROPOSITION_SET = "NON_UNIQUE_SUB_PROPOSITION_SET" 

class AtLeast(puan.Proposition):

    """
        :class:`AtLeast` proposition is in its core a regular at least expression (e.g. :math:`x+y+z \\ge 1`), but restricted
        to having only +1 or -1 as variable coefficients. This is set by the ``sign`` parameter. An :class:`AtLeast` proposition
        is considered invalid if there are no sub propositions given.

        Parameters
        ----------
        value : integer value constraint constant - right hand side of the inequality
        propositions : a list of :class:`puan.Proposition` instances or ``str``
        variable : variable connected to this proposition
        sign : marking wheather coefficients before each sub proposition are 1 (``Sign.POSITIVE``) or -1 (``Sign.NEGATIVE``).

        Notes
        -----
            - ``propositions`` may take on any integer value given by their inner bounds, i.e. they are not restricted to boolean values.
            - ``propositions`` list cannot be empty.
            - ``variable`` may be of type :class:`puan.variable`, ``str`` or ``None``.

                - If ``str``, a default :class:`puan.variable` will be constructed with its id=variable.
                - If ``None``, then an id will be generated based on its ``propositions``, ``value`` and ``sign``.

        Raises
        ------
            Exception
                | If ``sign`` is not -1 or 1.
                | If ``propositions`` is either empty or None.
                | If :class:`variable.bounds<puan.Bounds>` is not (0, 1).

        Examples
        --------
        At least one of x, y and z.
            >>> AtLeast(1, list("xyz"), variable="A")
            A: +(x,y,z)>=1
        
        At least three of a, b and c.
            >>> AtLeast(value=3, propositions=list("abc"), variable="A")
            A: +(a,b,c)>=3
        
        At least eight of t. Notice the ``dtype`` on variable `t`
            >>> AtLeast(value=8, propositions=[puan.variable("t", dtype="int")], variable="A")
            A: +(t)>=8
        
        Methods
        -------
        assume
        atomic_propositions
        bounds
        compound_propositions
        equation_bounds
        errors
        evaluate
        evaluate_propositions
        flatten
        from_json
        from_short
        id
        is_contradiction
        is_tautology
        negate
        reduce
        solve
        to_b64
        to_ge_polyhedron
        to_json
        to_short
        to_text
        variables

    """
    
    def __init__(self, value: int, propositions: typing.List[typing.Union[str, puan.Proposition]], variable: typing.Union[str, puan.variable] = None, sign: puan.Sign = None):
        self.generated_id = False
        self.value = value
        self.sign = sign
        if sign is None:
            self.sign = puan.Sign.POSITIVE if value > 0 else puan.Sign.NEGATIVE

        if not self.sign in [-1,1]:
            raise Exception(f"`sign` of AtLeast proposition must be either -1 or 1, got: {sign}")


        if propositions is None:
            raise Exception("Sub propositions cannot be `None`")

        propositions_list = list(propositions)
        self.propositions = sorted(
            itertools.chain(
                filter(
                    lambda x: type(x) != str, 
                    propositions_list
                ),
                map(
                    puan.variable,
                    filter(
                        lambda x: type(x) == str,
                        propositions_list
                    )
                )
            ),
        )

        if variable is None:
            self.variable = puan.variable(id=AtLeast._id_generator(self.propositions, value, sign))
            self.generated_id = True
        elif type(variable) == str:
            self.variable = puan.variable(id=variable, bounds=(0,1))
        elif issubclass(variable.__class__, puan.variable):
            if not variable.bounds.as_tuple() in [(0,0), (0,1), (1,1)]:
                raise ValueError(f"variable of a compound proposition cannot have bounds other than (0, 1), got: {variable.bounds}")
            self.variable = variable
        else:
            raise ValueError(f"`variable` must be of type `str` or an instance of `puan.variable`, got: {type(variable)}")

    def __repr__(self) -> str:
        atoms = sorted(
            map(lambda x: x.id, self.propositions)
        )
        return f"{self.variable.id}: {'+' if self.sign > 0 else '-'}({','.join(atoms)})>={self.value}"

    def __lt__(self, other):
        return self.id < other.id

    def __eq__(self, other):
        if not type(self) == type(other):
            return False

        return (self.id == other.id) & (self.equation_bounds == other.equation_bounds) & (self.value == other.value)

    def __hash__(self):
        return hash((self.variable, self.sign, self.value, tuple(self.propositions)))

    def _id_generator(propositions, value: int, sign: int, prefix: str = "VAR") -> str:
        return prefix + hashlib.sha256(
            str(
                "".join(
                    itertools.chain(
                        map(
                            operator.attrgetter("id"), 
                            filter(lambda x: issubclass(x.__class__, puan.variable), propositions)
                        ),
                        map(
                            operator.attrgetter("variable.id"), 
                            filter(lambda x: not issubclass(x.__class__, puan.variable), propositions)
                        )
                    )
                ) + str(value) + str(sign)
            ).encode()
        ).hexdigest()

    @property
    def id(self) -> str:
        """
            Id of this proposition.

            Returns
            -------
                out : str
        """
        return self.variable.id

    @property
    def bounds(self) -> puan.Bounds:
        """
            Variable bounds of this proposition.

            Returns
            -------
                out : :class:`puan.Bounds`
        """
        return self.variable.bounds

    @property
    def compound_propositions(self) -> typing.Iterable[puan.Proposition]:
        """
            All propositions of (inheriting) type :class:`AtLeast`.

            Examples
            --------
                >>> proposition = AtLeast(value=1,
                ...     propositions=["a", AtLeast(value=2, propositions=list("xy"), variable="B")],
                ...     variable="A")
                >>> list(proposition.compound_propositions)
                [B: +(x,y)>=2]
            
            See also
            --------
                atomic_propositions
            
            Returns
            -------
                out : Iterable[puan.Proposition]
        """
        return filter(lambda x: not issubclass(x.__class__, puan.variable), self.propositions)

    @property
    def atomic_propositions(self) -> typing.Iterable[puan.variable]:
        """
            All propositions of type :class:`puan.variable`.

            Examples
            --------
                >>> proposition = AtLeast(value=1,
                ...     propositions=["a", AtLeast(value=2, propositions=list("xy"), variable="B")],
                ...     variable="A")
                >>> list(proposition.atomic_propositions)
                [variable(id='a', bounds=Bounds(lower=0, upper=1))]
            
            See also
            --------
                compound_propositions
            
            Returns
            -------
                out : Iterable[:class:`puan.variable`]
        """
        return filter(lambda x: issubclass(x.__class__, puan.variable), self.propositions)

    @property
    def variables(self) -> typing.List[puan.variable]:

        """
            All variables in this proposition (including sub propositions).

            Returns
            -------
                out : List[:class:`puan.variable`]
        """
        return sorted(
            set(
                itertools.chain(
                    map(
                        operator.attrgetter('id'),
                        self.flatten(),
                    ),
                )
            )
        )

    def flatten(self) -> typing.List[puan.Proposition]:

        """
            Returns all its propositions and their sub propositions as a unique list of propositions.

            Examples
            --------
                >>> proposition = AtLeast(1, [AtLeast(1, ["a", "b"], "B"),
                ... AtLeast(1, ["c", "d"], "C"), "e"], "A")
                >>> proposition.flatten()
                [A: +(B,C,e)>=1, B: +(a,b)>=1, C: +(c,d)>=1, variable(id='a', bounds=Bounds(lower=0, upper=1)), variable(id='b', bounds=Bounds(lower=0, upper=1)), variable(id='c', bounds=Bounds(lower=0, upper=1)), variable(id='d', bounds=Bounds(lower=0, upper=1)), variable(id='e', bounds=Bounds(lower=0, upper=1))]
        
            Returns
            -------
                out : List[puan.Proposition]
        """

        return sorted(
            set(
                itertools.chain(
                    [self],
                    *map(
                        operator.methodcaller("flatten"),
                        self.compound_propositions
                    ),
                    self.atomic_propositions
                )
            )
        )

    def _dependencies(self) -> typing.List[typing.Tuple[puan.variable, typing.List[puan.variable]]]:

        """
            Returns a dependency graph from each proposition to it's
            sub propositions.

            Examples
            --------
                >>> All(*"abc", variable="A")._dependencies()
                [('A', ['a', 'b', 'c'])]

            Returns
            ------- 
                out : List[Tuple[:class:`puan.variable`, List[:class:`puan.variable`]]]
        """
        return list(
            itertools.chain(
                [
                    (
                        self.variable.id,
                        list(
                            itertools.chain(
                                map(
                                    operator.attrgetter("id"),
                                    self.atomic_propositions,
                                ),
                                map(
                                    operator.attrgetter("id"),
                                    self.compound_propositions
                                )
                            )   
                        )
                    )
                ],
                functools.reduce(
                    lambda a,b: a+b, 
                    map(
                        operator.methodcaller("_dependencies"),
                        self.compound_propositions
                    ),
                    []
                )
            )
        )

    def errors(self) -> typing.List[PropositionValidationError]:

        """
            Checks this proposition and returns a list of :class:`PropositionValidationError`'s. 
            An error in the result is due to one or more of the following:
            
                - It exist at least one circular variable dependency.
                - A variable depends directly on two or more variables with identical ids.
                - There exists two or more variables with identical ids but with different bounds.

            Examples
            --------
                >>> All(*"xyz", variable="A").errors()
                []

                >>> All(*"A", variable="A").errors()
                [<PropositionValidationError.CIRCULAR_DEPENDENCIES: 'CIRCULAR_DEPENDENCIES'>]

                >>> All(Any(*"ab", variable="A"), Any(puan.variable("A", (-10,10)))).errors()
                [<PropositionValidationError.AMBIVALENT_VARIABLE_DEFINITIONS: 'AMBIVALENT_VARIABLE_DEFINITIONS'>]

            Returns
            -------
                out : List[:class:`PropositionValidationError`]
        """
        return list(
            maz.compose(
                functools.partial(
                    itertools.compress,
                    [
                        PropositionValidationError.CIRCULAR_DEPENDENCIES,
                        PropositionValidationError.AMBIVALENT_VARIABLE_DEFINITIONS,
                        PropositionValidationError.AMBIVALENT_VARIABLE_DEFINITIONS,
                        PropositionValidationError.NON_UNIQUE_SUB_PROPOSITION_SET
                    ],
                ),
                maz.fnmap(

                    # Checks that if there exists any circlular
                    # variable dependecy somewhere in the model
                    maz.fnexcept(
                        maz.compose(
                            operator.not_,
                            functools.partial(
                                operator.eq,
                                None,
                            ),
                            operator.methodcaller("prepare"),
                            graphlib.TopologicalSorter,
                            dict,
                            operator.methodcaller("_dependencies")
                        ),
                        lambda _: True,
                    ),

                    # Checks that every node's variable with some id also has
                    # the same bounds. Otherwise, there is an ambivalent variable definition.
                    maz.compose(
                        operator.not_,
                        functools.partial(
                            maz.invoke,
                            operator.eq,   
                        ),
                        maz.fnmap(
                            maz.compose(
                                len,
                                set,
                                functools.partial(map, hash),
                                itertools.chain.from_iterable,
                                maz.fnmap(
                                    functools.partial(
                                        filter,
                                        lambda x: issubclass(
                                            x.__class__, 
                                            puan.variable
                                        ),
                                    ),
                                    maz.compose(
                                        functools.partial(
                                            map,
                                            operator.attrgetter("variable"),
                                        ),
                                        functools.partial(
                                            filter,
                                            lambda x: not issubclass(
                                                x.__class__, 
                                                puan.variable
                                            ),
                                        )
                                    )
                                ),
                                operator.methodcaller("flatten")
                            ),
                            maz.compose(
                                len,
                                set,
                                functools.partial(
                                    map, 
                                    operator.attrgetter("id")
                                ),
                                operator.methodcaller("flatten")
                            ),
                        )
                    ),

                    # Checks only compound propositions if there exist
                    # two or more that share id, bounds but not rest of
                    # a compound proposition properties
                    maz.compose(
                        operator.not_,
                        functools.partial(
                            maz.invoke,
                            operator.eq,   
                        ),
                        maz.fnmap(
                            maz.compose(len, set, functools.partial(map, hash)),
                            maz.compose(len, set, functools.partial(map, operator.attrgetter("id")))
                        ),
                        list,
                        functools.partial(
                            filter,
                            lambda x: not issubclass(x.__class__, puan.variable),
                        ),
                        operator.methodcaller("flatten")
                    ),


                    # Checks if any edge exists more than once
                    # E.g. A has two edges to x, meaning two x's are siblings
                    # which don't make any sense
                    maz.compose(
                        any,
                        functools.partial(
                            map,
                            lambda i: i >= 2,
                        ),
                        dict.values,
                        Counter,
                        itertools.chain.from_iterable,
                        functools.partial(
                            map, 
                            lambda x: list(
                                map(
                                    lambda y: f"{x.id}-{y.id}",
                                    x.propositions
                                )
                            )
                        ),
                        functools.partial(
                            filter,
                            lambda x: not issubclass(x.__class__, puan.variable)
                        ),
                        operator.methodcaller("flatten")
                    )
                )
            )(self)
        )

    def _to_pyrs_theory(self) -> typing.Tuple[pr.TheoryPy, typing.Dict[str, tuple]]:

        """
            Converts this plog model into a puan-rspy Theory.

            Returns
            -------
                out : Tuple[:class:`pr.TheoryPy`, Dict[``str``, ``tuple``])
        """
        flatten = self.flatten()
        flatten_dict = dict(zip(map(lambda x: x.id, flatten), flatten))
        variable_id_map = dict(
            zip(
                map(
                    lambda x: x.id, 
                    flatten_dict.values()
                ), 
                zip(
                    range(len(flatten_dict)), 
                    flatten_dict.values()
                )
            )
        )
        return pr.TheoryPy(
            list(
                map(
                    lambda x: pr.StatementPy(
                        variable_id_map[x.id][0],
                        variable_id_map[x.id][1].bounds.as_tuple(),
                        pr.AtLeastPy(
                            list(
                                map(
                                    lambda y: variable_id_map[y.id][0], 
                                    x.propositions
                                )
                            ),
                            bias=-1*x.value,
                            sign=pr.SignPy.Positive if x.sign == puan.Sign.POSITIVE else pr.SignPy.Negative
                        ) if not issubclass(x.__class__, puan.variable) else None,
                    ),
                    flatten_dict.values()
                )
            )
        ), variable_id_map

    def to_ge_polyhedron(self, active: bool = False, reduced: bool = False) -> pnd.ge_polyhedron:

        """
            Converts into a :class:`ge_polyhedron<puan.ndarray.ge_polyhedron>`.

            Parameters
            ----------
                active : bool = False
                    If true, then the top node id will be assumed to be true.

                reduced : bool = False
                    If true, then methods will be applied to the polyhedron in an attempt to reduce its size. 

            Returns
            -------
                out : :class:`puan.ndarray.ge_polyhedron`
        """

        flatten = self.flatten()
        flatten_dict = dict(zip(map(lambda x: x.id, flatten), flatten))
        variable_id_map = dict(
            zip(
                map(
                    lambda x: x.id, 
                    flatten_dict.values()
                ), 
                zip(
                    range(len(flatten_dict)), 
                    flatten_dict.values()
                )
            )
        )
        polyhedron_rs = pr.TheoryPy(
            list(
                map(
                    lambda x: pr.StatementPy(
                        variable_id_map[x.id][0],
                        variable_id_map[x.id][1].bounds.as_tuple(),
                        pr.AtLeastPy(
                            list(
                                map(
                                    lambda y: variable_id_map[y.id][0], 
                                    x.propositions
                                )
                            ),
                            bias=-1*x.value,
                            sign=pr.SignPy.Positive if x.sign == puan.Sign.POSITIVE else pr.SignPy.Negative
                        ) if not issubclass(x.__class__, puan.variable) else None,
                    ),
                    flatten_dict.values()
                )
            )
        ).to_ge_polyhedron(active, reduced)

        id_variable_map = dict(variable_id_map.values())
        polyedron_array = np.hstack(
            (
                np.array(polyhedron_rs.b).reshape(-1,1), 
                np.array(np.array_split(polyhedron_rs.a.val, polyhedron_rs.a.nrows))
            )
        )
        return pnd.ge_polyhedron(
            polyedron_array, 
            variables=[puan.variable.support_vector_variable()]+list(
                map(
                    maz.compose(
                        id_variable_map.get,
                        operator.attrgetter("id")
                    ), 
                    polyhedron_rs.variables
                )
            ),
        )

    def negate(self) -> puan.Proposition:

        """
            Negates proposition.

            Examples
            --------
                >>> AtLeast(2, ["x","y","z"], variable="A").negate()
                A: -(x,y,z)>=-1

                >>> AtLeast(3, ["x","y","z"], variable="A").negate()
                A: -(x,y,z)>=-2

                >>> AtLeast(-1, ["x","y","z"], variable="A", sign=puan.Sign.NEGATIVE).negate()
                A: +(x,y,z)>=2

            Returns
            -------
                out : :class:`puan.Proposition`
        """
        negated = AtLeast(
            value=(self.value*-1)+1,
            propositions=self.propositions,
            variable=None if self.generated_id else self.variable,
            sign=-1*self.sign,
        )
        
        # If negated went from positive to negative,
        # then move negation inwards into each compound
        # proposition. If no compound, then return as is
        atoms = list(negated.atomic_propositions)
        if (negated.sign == -1) and (len(atoms) < len(negated.propositions)):

            compounds = list(negated.compound_propositions)
            
            # If both compounds and atomics are less
            # than full len of propositions, then this
            # is a mixed of both
            if len(compounds) < len(self.propositions):
                compounds.append(
                    AtLeast(
                        value=self.value,
                        propositions=atoms,
                        sign=self.sign,
                    )
                )

            negated.propositions = list(
                map(
                    operator.methodcaller("negate"),
                    compounds
                ),
            )
            negated.sign = 1
            negated.value += len(compounds)

        return negated

    @property
    def _equation_mm(self) -> typing.Tuple[int, int]:

        """Max min value of equation exclusive bias"""
        can_min_val = sum(map(lambda x: min(x.bounds.as_tuple())*self.sign, self.propositions))
        can_max_val = sum(map(lambda x: max(x.bounds.as_tuple())*self.sign, self.propositions))
        min_val = min(can_min_val, can_max_val)
        max_val = max(can_min_val, can_max_val)
        return (min_val, max_val)

    @property
    def equation_bounds(self) -> typing.Tuple[int, int]:
        """
            The range of values the left hand side of equations on the form :math:`ax + by + cz - value(bias) \\ge 0` can take.

            Examples
            --------
                >>> model = AtLeast(1,["x","y"])
                >>> model.equation_bounds
                (-1, 1)

                >>> model = AtLeast(2,["x","y"])
                >>> model.equation_bounds
                (-2, 0)

                >>> model = AtMost(1,["x","y"])
                >>> model.equation_bounds
                (-1, 1)

                >>> model = AtMost(2,["x","y"])
                >>> model.equation_bounds
                (0, 2)

            Returns
            -------
                out : Tuple[int, int]
        """
        mn, mx = self._equation_mm
        return (mn-self.value, mx-self.value)

    @property
    def is_tautology(self) -> bool:

        """
            Returns wheather or not this proposition is true, no matter the interpretation of its propositions.

            Notes
            -----
            Sub propositions are not taken into consideration.

            Examples
            --------
                >>> model = AtLeast(1,["x","y"])
                >>> model.is_tautology
                False

                >>> model = AtMost(1,["x","y"])
                >>> model.is_tautology
                False

                >>> model = AtMost(3,["x","y","z"])
                >>> model.is_tautology
                True

                >>> model = AtLeast(0, ["x"], sign=1)
                >>> model.is_tautology
                True

                >>> model = AtMost(2,["x","y"])
                >>> model.is_tautology
                True

            Returns
            -------
                out : bool
        """
        # When the lowest sum from equation is still higher than value, this is a tautologi
        return self.equation_bounds[0] >= 0

    @property
    def is_contradiction(self) -> bool:

        """
            Returns wheather or not this proposition is false, no matter the interpretation of its propositions.

            Notes
            -----
            Sub propositions are not taken into consideration.

            Examples
            --------
                >>> model = AtLeast(1,["x","y"])
                >>> model.is_contradiction
                False

                >>> model = AtMost(1,["x","y"])
                >>> model.is_contradiction
                False

                >>> model = AtLeast(3,["x","y"])
                >>> model.is_contradiction
                True

                >>> model = AtMost(-1,["x","y"])
                >>> model.is_contradiction
                True
            

            Returns
            -------
                out : bool
        """
        # When the highest sum from equation still not satisfied inequality, this is a contradition
        return self.equation_bounds[1] < 0

    def evaluate(self, interpretation: typing.Dict[str, typing.Union[puan.Bounds, typing.Tuple[int,int], int]]) -> puan.Bounds:

        """
            Evaluates ``interpretation`` on this model. It will evaluate sub propositions
            bottoms-up and propagate results upwards. This means that even though 
            intermediate variables are not set in ``interpretation``, they (may) receive a new bound
            based on the evaluation of its ``propositions``.

            Parameters
            ----------
                interpretation : Dict[str, Union[puan.Bounds, Tuple[int,int], int]]
                    the values of the variables in the model to evaluate it for

            Examples
            --------
                >>> All(*"xy", variable="A").evaluate({"x": 1})
                Bounds(lower=0, upper=1)

                >>> All(*"xy", variable="A").evaluate({"x": (1,1)})
                Bounds(lower=0, upper=1)

                >>> All(*"xy", variable="A").evaluate({"x": puan.Bounds(1,1)})
                Bounds(lower=0, upper=1)

                >>> All(*"xy", variable="A").evaluate({"x": 1, "y": 1})
                Bounds(lower=1, upper=1)

                >>> AtLeast(propositions=[puan.variable("x", dtype="int")],
                ... value=10).evaluate({"x": 9})
                Bounds(lower=0, upper=0)

                >>> AtLeast(propositions=[puan.variable("x", dtype="int")],
                ... value=10).evaluate({"x": 10})
                Bounds(lower=1, upper=1)
            
            See also
            --------
                evaluate_propositions : Evaluates propositions on this model given a ``dict`` with variables and their bounds/constants.
            
            Returns
            -------
                out : bool
        """

        return self.evaluate_propositions(interpretation)[self.variable.id]

    def evaluate_propositions(
        self, 
        interpretation: typing.Dict[str, typing.Union[puan.Bounds, typing.Tuple[int,int], int]],
        out: typing.Callable[[puan.Bounds], typing.Union[puan.Bounds, typing.Tuple[int,int], int]] = lambda x: x,
    ) -> typing.Dict[str, puan.Bounds]:
        """
            Evaluates propositions on this model given a ``dict`` with variables and their bounds/constants.

            Parameters
            ----------
                interpretation : Dict[Union[str, :class:`puan.variable`], int]
                    the values of the variables in the model to evaluate it for
                out : Callback[[puan.Bounds], Union[puan.Bounds, Tuple[int,int], int]]
                    an optional callback function for changing output data type.

            Notes
            -----
                A variable's new bound, which is not included in the initial dict, is calculated from its sub proposition's bounds or
                kept unchanged. 

            Examples
            --------
                >>> All(*"xy", variable="A").evaluate_propositions({"x": 1})
                {'A': Bounds(lower=0, upper=1), 'x': Bounds(lower=1, upper=1), 'y': Bounds(lower=0, upper=1)}

                >>> All(*"xy", variable="A").evaluate_propositions({"x": 1, "y": 1})
                {'A': Bounds(lower=1, upper=1), 'x': Bounds(lower=1, upper=1), 'y': Bounds(lower=1, upper=1)}

                >>> All(*"xy", variable="A").evaluate_propositions({"x": 1, "y": (1,1)})
                {'A': Bounds(lower=1, upper=1), 'x': Bounds(lower=1, upper=1), 'y': Bounds(lower=1, upper=1)}

                >>> All(*"xy", variable="A").evaluate_propositions({"x": puan.Bounds(1,1), "y": (1,1)})
                {'A': Bounds(lower=1, upper=1), 'x': Bounds(lower=1, upper=1), 'y': Bounds(lower=1, upper=1)}

                >>> All(*"xy", variable="A").evaluate_propositions({"x": puan.Bounds(1,1), "y": (1,1)}, out=lambda x: x.constant)
                {'A': 1, 'x': 1, 'y': 1}

                >>> AtLeast(propositions=[puan.variable("x", (0, 10))],
                ... value=10).evaluate_propositions({"x": 9})
                {'VAR2fa10da7075e3abf61065cb37ecd6bb658b38c9fdd0a1b1a69e34d541d32bd2d': Bounds(lower=0, upper=0), 'x': Bounds(lower=9, upper=9)}

                >>> AtLeast(propositions=[puan.variable("x", dtype="int")],
                ... value=10).evaluate_propositions({"x": 10})
                {'VAR2fa10da7075e3abf61065cb37ecd6bb658b38c9fdd0a1b1a69e34d541d32bd2d': Bounds(lower=1, upper=1), 'x': Bounds(lower=10, upper=10)}
            
            See also
            --------
                evaluate : Evaluates interpretation on this model.

            Returns
            -------
                out : Dict[str, Union[puan.Bounds, Tuple[int,int], int]]
        """
        return dict(
            zip(
                *maz.fnmap(
                    functools.partial(
                        map,
                        operator.attrgetter("id"),
                    ),
                    functools.partial(
                        map,
                        maz.compose(
                            out,
                            operator.attrgetter("bounds")
                        ),
                    ),
                )(self.assume(interpretation).flatten())
            )
        )

    def to_short(self) -> typing.Tuple[str, int, typing.List[str], int, typing.List[int]]:

        """
            `short` is a tuple format with five element types:
            (0) id, (1) sign, (2) propositions, (3) bias (4) bounds.

            Notes
            -----
                :meth:`to_short` does not include sub propositions if any

            Examples
            --------
                >>> All("x","y","z",variable="A").to_short()
                ('A', 1, ['x', 'y', 'z'], -3, [0, 1])

            Returns
            -------
                out : Tuple[str, int, List[str], int, List[int]]
        """
        return (self.id, int(self.sign), list(map(operator.attrgetter("id"), self.propositions)), -1*self.value, list(self.bounds.as_tuple()))

    def to_text(self) -> str:

        """
            Returns a readable, concise, version controllable text format.

            Returns
            -------
                out : str
        """
        return "\n".join(
            sorted(
                map(
                    maz.compose(
                        str,
                        operator.methodcaller("to_short")
                    ),
                    set(self.flatten())
                ),
            )
        ).replace(" ", "")

    def to_json(self) -> typing.Dict[str, typing.Any]:

        """
            Returns proposition as a readable JSON.

            Examples
            --------
                >>> All("x","y","z",variable="A").to_json()
                {'type': 'All', 'propositions': [{'id': 'x'}, {'id': 'y'}, {'id': 'z'}], 'id': 'A'}

            Returns
            -------
                out : Dict[str, Any]
        """
        d = {
            'type': self.__class__.__name__,
            'propositions': list(
                map(
                    operator.methodcaller("to_json"),
                    self.propositions
                )
            ),
            'value': self.value
        }
        if not self.generated_id:
            d['id'] = self.id

        return d

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
    
    @staticmethod
    def from_json(data: typing.Dict[str, typing.Any], class_map: list) -> "AtLeast":

        """
            Convert from JSON data to a proposition.

            Parameters
            ----------
                class_map : list
                    A list of classes implementing the ``puan.Proposition`` protocol.
                    They'll be mapped from `type` attribute in the json data.

            Notes
            -----
            Propositions within data not having `type` attribute will be considered a :class:`puan.variable`.

            Examples
            --------
                >>> AtLeast.from_json({"type": "AtLeast", "propositions": [{"id":"x"},{"id":"y"},{"id":"z"}], "id": "A"}, [puan.variable, AtLeast])
                A: +(x,y,z)>=1

            Returns
            -------
                out : :class:`AtLeast`
        """
        propositions = data.get('propositions', [])
        return AtLeast(
            value=data.get('value', 1),
            propositions=list(map(functools.partial(from_json, class_map=class_map), propositions)),
            variable=data.get('id', None)
        )

    @staticmethod
    def from_short(short: typing.Tuple[str, puan.Sign, typing.List[str], int, typing.List[int]]) -> "AtLeast":

        """
            From short data format into an :class:`AtLeast` proposition.
            A short data format is a tuple of id, sign, variables, bias and bounds.

            Examples
            --------
                >>> AtLeast.from_short(("A", 1, ["a","b","c"], -1, [0,1]))
                A: +(a,b,c)>=1

                >>> AtLeast.from_short(("x", 1, [], 0, [-10,10]))
                variable(id='x', bounds=Bounds(lower=-10, upper=10))

            Raises
            ------
                Exception
                    If tuple is not of size 5.

            Returns
            -------
                out : :class:`AtLeast`
        """
        try:
            _id, sign, props, bias, bounds = short
        except Exception as e:
            raise Exception(f"tried to convert short into ``AtLeast`` propositions but failed due to: {e}")

        if len(props) > 0:
            return AtLeast(
                value=-1*bias,
                propositions=props,
                variable=puan.variable(_id, bounds),
                sign=sign,
            )
        else:
            return puan.variable(_id, bounds)

    def solve(
        self, 
        objectives: typing.List[typing.Dict[typing.Union[str, puan.variable], int]], 
        solver: typing.Callable[[pnd.ge_polyhedron, typing.Iterable[np.ndarray]], typing.Iterable[typing.Tuple[typing.Optional[np.ndarray], typing.Optional[int], int]]] = None,
        try_reduce_before: bool = False,
        include_virtual_variables: bool = False,
    ) -> itertools.starmap:

        """
            Maximises objective in objectives such that this model's constraints are fulfilled and returns a solution for each objective.

            Parameters
            ----------
                objectives : List[Dict[Union[str, :class:`puan.variable`], int]]
                    A list of objectives as dictionaries. Keys are either variable ids as strings or a :class:`puan.variable`.
                    Values are objective value for the key.

                solver : Callable[[:class:`puan.ndarray.ge_polyhedron`, Dict[str, int]], List[(:class:`np.ndarray`, int, int)]] = None
                    If None is provided puan's own (beta) solver is used. If you want to provide another solver
                    you have to send a function as solver parameter. That function has to take a :class:`puan.ndarray.ge_polyhedron` and
                    a 2d numpy array representing all objectives, as input. NOTE that the polyhedron DOES NOT provide constraints for variable
                    bounds. Variable bounds are found under each variable under `polyhedron.variables` and constraints for 
                    these has to manually be created and added to the polyhedron matrix. The function should return a list, one for each
                    objective, of tuples of (solution vector, objective value, status code). The solution vector is an integer ndarray vector
                    of size equal to width of ``polyhedron.A``. There are six different status codes from 1-6:
                    
                    - 1: solution is undefined
                    - 2: solution is feasible
                    - 3: solution is infeasible
                    - 4: no feasible solution exists
                    - 5: solution is optimal
                    - 6: solution is unbounded

                    Checkout https://github.com/ourstudio-se/puan-solvers for quick how-to's for common solvers.

                try_reduce_before : bool = False
                    If true, then methods will be applied to try and reduce size of this model before
                    running solve function.

                include_virtual_variables : bool = False
                    If true, the virtual/artificial variables that has automatically been generated creating the model will be included in solutions

            Examples
            --------
                >>> dummy_solver = lambda x,y: list(map(lambda v: (v, 0, 5), y))
                >>> list(All(*"ab").solve([{"a": 1, "b": 1}], dummy_solver))
                [({'a': 1, 'b': 1}, 0, 5)]

            Notes
            -----
                Currently a beta solver is used, *DO NOT USE THIS IN PRODUCTION*.
                If no solution could be found, ``None`` is returned.

            Returns
            -------
                out : :class:`itertools.starmap`:
        """

        ph = self.to_ge_polyhedron(
            active=True, 
            reduced=try_reduce_before,
        )
        if solver is None:
            pyrs_theory, variable_id_map = self._to_pyrs_theory()
            id_map = dict(variable_id_map.values())
            return itertools.starmap(
                lambda solution, objective_value, status_code: (
                    dict(
                        itertools.starmap(
                            lambda k,v: (id_map[k].id, v), 
                            filter(
                                maz.ifttt(
                                    # If is puan.variable
                                    lambda x: issubclass(id_map[x[0]].__class__, puan.variable),
                                    
                                    # then include it
                                    lambda _: True,

                                    # else if variable id is generated
                                    maz.ifttt(
                                        maz.compose(
                                            operator.attrgetter("generated_id"),
                                            id_map.get,
                                            operator.itemgetter(0),
                                        ),

                                        # then also check that we should include those variables
                                        lambda _: include_virtual_variables,

                                        # else include it
                                        lambda _: True
                                    )
                                ),
                                solution.items()
                            )
                        )
                    ),
                    objective_value, status_code
                ),
                pyrs_theory.solve(
                    list(
                        map(
                            lambda objective: dict(
                                zip(
                                    map(
                                        lambda k: variable_id_map[k][0],
                                        objective, 
                                    ),
                                    objective.values(),
                                )
                            ),
                            objectives, 
                        ),
                    ),
                    False,
                ),
            )
        else:
            polyhedron = self.to_ge_polyhedron(
                active=True, 
                reduced=try_reduce_before,
            )
            id_map = dict(
                zip(
                    range(polyhedron.A.shape[1]), 
                    polyhedron.A.variables
                )
            )
            return itertools.starmap(
                lambda solution, objective_value, status_code: (
                    dict(
                        map(
                            lambda x: (x[0].id, x[1]),
                            filter(
                                maz.ifttt(
                                    # if variable is a puan.variable
                                    lambda x: issubclass(x[0].__class__, puan.variable),
                                    
                                    # then keep it 
                                    lambda _: True,

                                    # else if variable id is generated
                                    maz.ifttt(
                                        maz.compose(
                                            operator.attrgetter("generated_id"),
                                            operator.itemgetter(0),
                                        ),

                                        # then check also that we should include those variables
                                        lambda _: include_virtual_variables,

                                        # else include it
                                        lambda _: True
                                    )
                                ),
                                zip(
                                    polyhedron.A.variables,
                                    solution
                                )
                            )
                        ),
                    ) if solution is not None else {},
                    objective_value, status_code
                ),
                solver(
                    polyhedron, 
                    map(polyhedron.A.construct, objectives),
                )
            )
            
    def reduce(self) -> puan.Proposition:

        """
            Reduces proposition by removing all variables with fixed bound.
            Returns a potentially reduced proposition.

            See also
            --------
                assume

            Examples
            --------
                >>> All(puan.variable("x", (1,1)), *"yz", variable="A").reduce()
                A: +(y,z)>=2
                >>> # Note that x is no longer part of the proposition and
                >>> # the value of the proposition is updated from 3 to 2
                >>> # as a result of x being 1

                >>> All(puan.variable("x", (1,1)), puan.variable("y", (0,0)), variable="A").reduce()
                variable(id='A', bounds=Bounds(lower=0, upper=0))
                >>> # If the remaining proposition is a puan.variable and only a puan.varialbe it is kept with updated bounds
                >>> # Any other constant proposition will be reduced

            Returns
            -------
                out :  :class:`puan.Proposition`
        """
        if self.bounds.constant is not None:
            return self.variable

        sub_propositions = list(
            itertools.chain(
                map(
                    operator.methodcaller("reduce"),
                    self.compound_propositions,
                ),
                self.atomic_propositions,
            ),
        )

        # it may occur that a proposition's new sub proposition list results in a new bound for
        # this proposition to be constant. Because of this we calculate the new bounds
        # and reduce the proposition if possible. 
        new_bounds = puan.Bounds(
            *(
                np.array(
                    list(
                        map(
                            # flip bounds and multiply by sign if sign is negative
                            # eg. if bounds (0,1) then (-1,0)
                            lambda x: x if self.sign > 0 else (x[1]*self.sign, x[0]*self.sign),
                            map(
                                lambda prop: prop.bounds.as_tuple(),
                                sub_propositions,
                            )
                        )
                    )
                ).sum(axis=0) >= self.value
            ) * 1
        )

        if new_bounds.constant is not None:
            return puan.variable(
                id=self.id,
                bounds=new_bounds,
            )
        
        return AtLeast(
            self.value - sum(
                map(
                    lambda x: x.bounds.constant, 
                    filter(
                        lambda x: x.bounds.constant is not None,
                        sub_propositions,
                    )
                )
            ) * self.sign,
            list(filter(lambda x: x.bounds.constant is None, sub_propositions)),
            variable=puan.variable(
                id=self.id,
                bounds=new_bounds,
            ),
            sign=self.sign,
        )

    def assume(self, new_variable_bounds: typing.Dict[str, typing.Union[int, typing.Tuple[int, int], puan.Bounds]]) -> puan.Proposition:

        """
            Assumes something about variable's bounds and returns a new proposition with these new bounds set.
            Other variables, not declared in ``new_variable_bounds``, may also get new bounds as a consequence from the ones set
            in ``new_variable_bounds``.

            Parameters
            ----------
                new_variable_bounds : typing.Dict[str, Union[int, Tuple[int, int], puan.Bounds]]
                    A dict of ids and either ``int``, ``tuple`` or :class:`puan.Bounds` as bounds for the variable.

            Notes
            -----
                All propositions are still kept within this proposition after assume. What may
                have happen is that proposition's bounds have tightened.

            See also
            --------
                reduce

            Examples
            --------
                Notice in this example that x bounds are changed from (0,1) to (1,1). Also that the
                model is a tautology after the assumptions.
                
                >>> model = Any(*"xy", variable="A")
                >>> model.propositions
                [variable(id='x', bounds=Bounds(lower=0, upper=1)), variable(id='y', bounds=Bounds(lower=0, upper=1))]
                >>> model.is_tautology
                False
                >>> model_assumed = model.assume({"x": 1}) # fixes x bounds to (1,1)
                >>> model_assumed.propositions
                [variable(id='x', bounds=Bounds(lower=1, upper=1)), variable(id='y', bounds=Bounds(lower=0, upper=1))]
                >>> model_assumed.is_tautology
                True

            Returns
            -------
                out : :class:`puan.Proposition`
        """
        if self.id in new_variable_bounds:
            self.variable = puan.variable(
                id=self.id,
                bounds=new_variable_bounds.get(self.id),
            )

        if self.bounds.constant is not None:
            return self.variable
        else:
            assumed_propositions = list(
                map(
                    lambda prop: prop.assume(new_variable_bounds),
                    self.propositions,
                )
            )

            result = AtLeast(
                value=self.value,
                propositions=maz.filter_map_concat(
                    # If proposition has a constant bound after evaluating it
                    lambda prop: prop.id in new_variable_bounds,
                    # If is a constant, just keep the variable from the proposition
                    # else, keep the proposition as is
                    lambda prop: prop if issubclass(prop.__class__, puan.variable) else prop.variable,
                )(assumed_propositions),
                variable=puan.variable(
                    self.id,
                    bounds=(
                        np.array(
                            list(
                                map(
                                    # flip bounds and multiply by sign if sign is negative
                                    # eg. if bounds (0,1) then (-1,0)
                                    lambda x: x if self.sign > 0 else (x[1]*self.sign, x[0]*self.sign),
                                    map(
                                        lambda prop: prop.bounds.as_tuple(),
                                        assumed_propositions,
                                    )
                                )
                            )
                        ).sum(axis=0) >= self.value
                    ) * 1,
                ),
                sign=self.sign,
            )
            result.__class__ = self.__class__
            return result


class AtMost(AtLeast):

    """
        ``AtMost`` proposition is an at least expression with negative coefficients and positive bias 
        (e.g. :math:`-x-y-z+1 \\ge 0`). Sub propositions may take on any value given by their equation bounds

        Parameters
        ----------
        value : integer value constraint constant - right hand side of the inequality
        propositions : a list of :class:`puan.Proposition` instances or ``str``
        variable : variable connected to this proposition

        Notes
        -----
            - Propositions may be of type ``str``, :class:`puan.variable` or :class:`AtLeast` (or other inheriting :class:`AtLeast`) 
            - Propositions list cannot be empty.

        Examples
        --------
        Meaning at most one of x, y and z.
            >>> AtMost(1, list("xyz"), variable="A")
            A: -(x,y,z)>=-1
        
        Methods
        -------
        from_json
        to_json
    """
    
    def __init__(self, value: int, propositions: typing.List[typing.Union[str, puan.variable]], variable: typing.Union[str, puan.variable] = None):
        super().__init__(value=-1*value, propositions=propositions, variable=variable, sign=puan.Sign.NEGATIVE)

    @staticmethod
    def from_json(data: dict, class_map) -> "AtMost":
        """
            Convert from JSON data to a proposition.

            Returns
            -------
                out : :class:`AtMost`
        """
        propositions = data.get('propositions', [])
        return AtMost(
            value=data.get('value', 1),
            propositions=list(map(functools.partial(from_json, class_map=class_map), propositions)),
            variable=data.get('id', None)
        )

    def to_json(self) -> typing.Dict[str, typing.Any]:

        """
            Returns proposition as a readable JSON.

            Returns
            -------
                out : Dict[str, Any]
        """
        d = super().to_json()
        d['value'] = -1*self.value
        return d

class All(AtLeast):

    """
        :class:`All` proposition means that all of its propositions must be true, otherwise the proposition
        is false.

        Parameters
        ----------
        *propositions : an iterable of :class:`puan.Proposition` instances or ``str``
        variable : variable connected to this proposition

        Notes
        -----
            - Propositions may be of type ``str``, :class:`puan.variable` or :class:`AtLeast` (or other inheriting :class:`AtLeast`) 
            - Propositions list cannot be empty.

        Examples
        --------
        Meaning at least one of x, y and z.
            >>> All(*"xyz", variable="A")
            A: +(x,y,z)>=3
        
        Methods
        -------
        from_json
        from_list
        to_json
    """
    
    def __init__(self, *propositions, variable: typing.Union[str, puan.variable] = None):
        super().__init__(value=len(set(propositions)), propositions=propositions, variable=variable)

    @staticmethod
    def from_json(data: typing.Dict[str, typing.Any], class_map) -> "All":
        """
            Convert from JSON data to a proposition.

            Returns
            -------
                out : :class:`All`
        """
        propositions = data.get('propositions', [])
        return All(
            *map(functools.partial(from_json, class_map=class_map), propositions),
            variable=data.get('id', None)
        )

    @staticmethod
    def from_list(propositions: list, variable: typing.Union[str, puan.variable] = None) -> "All":

        """
            Convert from list of propositions to an object of this proposition class.

            Examples
            --------
                >>> model = All.from_list([puan.variable("x"), puan.variable("y")], variable="a")
                >>> type(model) == All
                True

            Returns
            -------
                out : All
        """

        return All(*propositions, variable=variable)

    def to_json(self) -> typing.Dict[str, typing.Any]:

        """
            Returns proposition as a readable JSON.

            Returns
            -------
                out : Dict[str, Any]
        """
        d = super().to_json()
        del d['value']
        d['propositions'] = list(
            map(
                operator.methodcaller("to_json"),
                self.propositions
            )
        )
        return d

class Any(AtLeast):

    """
        :class:`Any` proposition means that at least one of its propositions must be true, otherwise the proposition
        is false.

        Parameters
        ----------
        *propositions : an iterable of :class:`puan.Proposition` instances or ``str``
        variable : variable connected to this proposition

        Notes
        -----
            - Propositions may be of type ``str``, :class:`puan.variable` or :class:`AtLeast` (or other inheriting :class:`AtLeast`) 
            - Propositions list cannot be empty.

        Examples
        --------
        Meaning at least one of x, y and z.
            >>> Any(*"xyz", variable="A")
            A: +(x,y,z)>=1
        
        Methods
        -------
        from_json
        from_list
        to_json
    """

    def __init__(self, *propositions, variable: typing.Union[str, puan.variable] = None):
        super().__init__(value=1, propositions=propositions, variable=variable)

    @staticmethod
    def from_json(data: typing.Dict[str, typing.Any], class_map) -> "Any":
        """
            Convert from JSON data to a proposition.

            Returns
            -------
                out : :class:`Any`
        """
        propositions = data.get('propositions', [])
        return Any(
            *map(functools.partial(from_json, class_map=class_map), propositions),
            variable=data.get('id', None)
        )

    @staticmethod
    def from_list(propositions: typing.List[typing.Union["AtLeast", puan.variable]], variable: typing.Union[str, puan.variable] = None) -> "Any":

        """
            Convert from list of propositions to an object of this proposition class.

            Examples
            --------
                >>> model = Any.from_list([puan.variable("x"), puan.variable("y")], variable="a")
                >>> type(model) == Any
                True

            Returns
            -------
                out : :class:`Any`
        """

        return Any(*propositions, variable=variable)

    def to_json(self) -> typing.Dict[str, typing.Any]:

        """
            Returns proposition as a readable JSON.

            Returns
            -------
                out : Dict[str, Any]
        """
        d = super().to_json()
        del d['value']
        d['propositions'] = list(
            map(
                operator.methodcaller("to_json"),
                self.propositions
            )
        )
        return d

class Imply(Any):

    """
        :class:`Imply` proposition consists of two sub propositions: condition and consequence. An implication
        proposition says that if the condition is true then the consequence must be true. Otherwise the proposition is false.

        Parameters
        ----------
        condition : an instance of :class:`puan.Proposition` data type or ``str``
        consequence : an instance of :class:`puan.Proposition` data type or ``str``
        variable : variable connected to this proposition

        Notes
        -----
            - Propositions may be of type ``str``, :class:`puan.variable` or :class:`AtLeast` (or other inheriting :class:`AtLeast`) 
            - Condition and consequence must be set.

        Examples
        --------
        Meaning at least one of x, y and z.
            >>> Imply(condition=All(*"abc", variable="B"), consequence=Any(*"xyz", variable="C"), variable="A")
            A: +(B,C)>=1
        
        Methods
        -------
        from_cicJE
        from_json
        to_json
    """

    def __init__(self, condition, consequence, variable: typing.Union[str, puan.variable] = None):
        if type(condition) == str or issubclass(condition.__class__, puan.variable):
            condition = All(condition)
        self.condition = condition.negate()
        self.consequence = consequence
        super().__init__(self.condition, self.consequence, variable=variable)

    @staticmethod
    def from_json(data: typing.Dict[str, typing.Any], class_map) -> "Imply":
        """
            Convert from JSON data to a proposition.

            Raises
            ------
                Exception
                    If no ``consequence`` key in ``data``.

            Returns
            -------
                out : :class:`Imply`
        """
        if not 'consequence' in data:
            raise Exception("type `Imply` must have a `consequence` proposition")

        if 'condition' in data and 'consequence' in data:
            return Imply(
                from_json(data.get('condition'), class_map),
                from_json(data.get('consequence'), class_map),
                variable=data.get('id', None)
            )
        else:
            return from_json(data.get('consequence'), class_map)

    def to_json(self) -> typing.Dict[str, typing.Any]:

        """
            Returns proposition as a readable JSON.

            Returns
            -------
                out : Dict[str, Any]
        """
        d = {
            'type': self.__class__.__name__,
            'condition': self.condition.negate().to_json(),
            'consequence': self.consequence.to_json(),
        }
        if not self.generated_id:
            d['id'] = self.id
        return d

    @staticmethod
    def from_cicJE(data: typing.Dict[str, typing.Any], id_ident: str = "id") -> "Imply":

        """
            This function converts a cicJE into an :class:`Imply`.
            
            Parameters
            ----------
            data : Dict[str, Any]
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
                someId: +(VAR180540a846781f231b3c1fb0422d95e48e9b9379a5ec6890a0b9a32cb7f66b75,VAR9b4a27f7babf7022c3cadd7e00de9f6da1747f82c5c95960f76138599c35b52c)>=1

            Returns
            -------
                out : :class:`Imply`
        """
        rule_type_map = {
            "REQUIRES_ALL": lambda x,id: All(*x,variable=id),
            "REQUIRES_ANY": lambda x,id: Any(*x,variable=id),
            "ONE_OR_NONE": lambda x,id: AtMost(*x,value=1,variable=id),
            "FORBIDS_ALL":  lambda x,id: Any(*x,variable=id).invert(),
            "REQUIRES_EXCLUSIVELY": lambda x,id: Xor(*x,variable=id)
        }
        cmp2prop = lambda x: puan.variable(id=x[id_ident], bounds=[(0,1),(puan.default_min_int, puan.default_max_int)]["dtype" in x and x['type'] == "int"])
        relation_fn = lambda x: [Any, All][x.get("relation", "ALL") == "ALL"]
        consequence = rule_type_map[data.get('consequence', {}).get("ruleType")](
            map(cmp2prop, data.get('consequence',{}).get('components')),
            data.get("consequence", {}).get("id", None)
        )
        if "condition" in data:
            condition_outer_cls = relation_fn(data['condition'])
            condition_inner_cls = list(map(
                lambda x: relation_fn(x)(*map(cmp2prop, x.get('components', [])), variable=x.get("id", None)),
                data['condition'].get("subConditions", [])
            ))
            if len(condition_inner_cls) > 0:
                return Imply(
                    condition=condition_outer_cls(
                        *condition_inner_cls, 
                        variable=data.get('condition', {}).get("id", None)
                    ) if len(condition_inner_cls) > 1 else condition_inner_cls[0],
                    consequence=consequence,
                    variable=data.get("id", None)
                )
            else:
                return consequence
        else:
            return consequence

class Xor(All):

    """
        :class:`Xor` proposition is true when exactly one of its propositions is true, e.g. :math:`x+y+z = 1`

        Parameters
        ----------
        *propositions : an iterable of :class:`puan.Proposition` instances or ``str``
        variable : variable connected to this proposition

        Notes
        -----
            - Propositions may be of type ``str``, :class:`puan.variable` or :class:`AtLeast` (or other inheriting :class:`AtLeast`) 
            - Propositions list cannot be empty.
        
        Methods
        -------
        from_json
        from_list
        to_json
    """

    def __init__(self, *propositions, variable: typing.Union[str, puan.variable] = None):
        super().__init__(
            AtLeast(value=1, propositions=propositions), 
            AtMost(value=1, propositions=propositions), 
            variable=variable,
        )

    @classmethod
    def from_json(cls, data: dict, class_map) -> "Xor":
        """
            Convert from JSON data to a proposition.

            Returns
            -------
                out : :class:`Xor`
        """
        propositions = data.get('propositions', [])
        return cls(
            *map(functools.partial(from_json, class_map=class_map), propositions),
            variable=data.get('id', None)
        )

    @classmethod
    def from_list(cls, propositions: typing.List[typing.Union["AtLeast", puan.variable]], variable: typing.Union[str, puan.variable] = None) -> "Xor":

        """
            Convert from list of propositions to an object of this proposition class.

            Examples
            --------
                >>> model = Xor.from_list([puan.variable("x"), puan.variable("y")], variable="a")
                >>> type(model) == Xor
                True

            Returns
            -------
                out : :class:`Xor`
        """

        return cls(*propositions, variable=variable)

    def to_json(self) -> typing.Dict[str, typing.Any]:

        """
            Returns proposition as a readable JSON.

            Returns
            -------
                out : Dict[str, Any]
        """
        d = {
            'type': self.__class__.__name__,
            'propositions': list(
                map(
                    operator.methodcaller("to_json"),
                    self.propositions[0].propositions
                )
            ) if len(self.propositions) > 0 else [],
        }
        if not self.generated_id:
            d['id'] = self.id
        return d


class ExactlyOne(Xor):    
    pass


class Not():

    """
        ``Not`` proposition negates the input proposition. 

        Parameters
        ----------
        proposition : an instance of :class:`puan.Proposition` or ``str``

        Notes
        -----
            - Proposition may be of type ``str``, :class:`puan.variable` or :class:`AtLeast` (or other inheriting :class:`AtLeast`)
        
        Methods
        -------
        from_json
    """

    def __new__(self, proposition):
        return (All(proposition) if type(proposition) == str or issubclass(proposition.__class__, puan.variable) else proposition).negate()

    @staticmethod
    def from_json(data: typing.Dict[str, typing.Any], class_map) -> "AtLeast":
        """
            Convert from JSON data to a proposition.

            Raises
            ------
                Exception
                    If no ``proposition`` key in ``data``.

            Returns
            -------
                out : :class:`AtLeast`
        """
        if not 'proposition' in data:
            raise Exception("type `Not` expects field `proposition` to have a proposition set")

        return Not(
            from_json(data['proposition'], class_map=class_map),
        )

class XNor(Any):

    """
        :class:`XNor` proposition is a negated :class:`Xor`, i.e. "exactly one" configurations will be false or :math:`x+y+z\\neq1`. 

        Parameters
        ----------
        *propositions : an iterable of :class:`puan.Proposition` instances or ``str``
        variable : variable connected to this proposition

        Notes
        -----
            - Proposition may be of type ``str``, :class:`puan.variable` or :class:`AtLeast` (or other inheriting :class:`AtLeast`) 
        
        Methods
        -------
        from_json
        from_list
        to_json
    """

    def __init__(self, *propositions, variable: typing.Union[puan.variable, str] = None):
        super().__init__(
            AtLeast(value=1, propositions=propositions).negate(), 
            AtMost(value=1, propositions=propositions).negate(), 
            variable=variable,
        )

    @staticmethod
    def from_json(data: dict, class_map) -> "XNor":
        """
            Convert from JSON data to a proposition.

            Returns
            -------
                out : :class:`AtLeast`
        """
        propositions = data.get('propositions', [])
        return XNor(
            *map(functools.partial(from_json, class_map=class_map), propositions),
            variable=data.get('id', None)
        )

    @staticmethod
    def from_list(propositions: list, variable: typing.Union[str, puan.variable] = None) -> AtLeast:

        """
            Convert from list of propositions to an object of this proposition class.

            Notes
            -----    
                :class:`XNor` is not its own type but instead returns an :class:`AtLeast` proposition which preserves the same logic.

            Examples
            --------
                >>> model = XNor.from_list([puan.variable("x"), puan.variable("y")], variable="a")
                >>> type(model) == XNor
                True

            Returns
            -------
                out : :class:`AtLeast`
        """

        return XNor(*propositions, variable=variable)

    def to_json(self) -> typing.Dict[str, typing.Any]:

        """
            Returns proposition as a readable JSON.

            Returns
            -------
                out : Dict[str, Any]
        """
        d = {
            'type': self.__class__.__name__,
            'propositions': list(
                map(
                    maz.compose(operator.methodcaller("to_json")),
                    self.propositions[0].negate().propositions
                )
            ) if len(self.propositions) > 0 else [],
        }
        if not self.generated_id:
            d['id'] = self.id
        return d

def from_json(data: dict, class_map: list = [puan.variable,AtLeast,AtMost,All,Any,Xor,ExactlyOne,Not,XNor,Imply]) -> typing.Any:

    """
        Convert from json data to a proposition.

        Raises
        ------
            Exception
                If no parse function for `type` value is not implemented.

        Returns
        -------
            out : typing.Any
    """
    _class_map = dict(zip(map(lambda x: x.__name__, class_map), class_map))
    if 'type' not in data:
        if 'propositions' in data:
            return _class_map["AtLeast"].from_json(data, class_map)
        else:
            return _class_map["variable"].from_json(data, class_map)
    elif data['type'] in ["Proposition", "Variable"]:
        return _class_map["variable"].from_json(data, class_map)
    elif data['type'] in _class_map:
        return _class_map[data['type']].from_json(data, class_map)
    else:
        raise Exception(f"cannot parse type '{data['type']}'")

def from_b64(base64_str: str) -> typing.Any:

    """
        Unpacks base64 string `base64_str` into some data.

        Parameters
        ----------
            base64_str: str

        Raises
        ------
            Exception
                If error occurred while decompressing.

        Returns
        -------
            out : Any
    """
    try:
        return pickle.loads(
            gzip.decompress(
                base64.b64decode(
                    base64_str.encode()
                )
            )
        )
    except:
        raise Exception("could not decompress and load polyhedron from string: version mismatch.")
