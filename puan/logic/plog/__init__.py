import hashlib
import maz
import base64
import gzip
import pickle
import functools
import itertools
import typing
import operator
import numpy as np
import puan
import puan.ndarray as pnd
import puan_rspy as pst
import dictdiffer
import more_itertools
from toposort import toposort
from dataclasses import dataclass

class AtLeast(puan.StatementInterface):

    """
        ``AtLeast`` proposition is in its core a regular at least expression (e.g. :math:`x+y+z \ge 1`), but restricted
        to having only +1 or -1 as variable coefficients. This is set by the `sign` property. An ``AtLeast`` proposition
        is considered invalid if there are no sub propositions given.

        Notes
        -----
            - `propositions` may take on any integer value given by their inner bounds, i.e. they are not restricted to boolean values.
            - `propositions` list cannot be empty.
            - `sign` parameter take only -1 or 1 as value.
            - `variable` may be of type ``puan.variable``, ``str`` or ``None``. 
                - If ``str``, a default ``puan.variable`` will be constructed with its id=variable.
                - If ``None``, then an id will be generated based on its propositions, value and sign.

        Raises
        ------
            Exception
                | If `sign` is not -1 or 1.
                | If `propositions` is either empty or None.
                | If variable bounds is not (0, 1).

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
    """
    
    def __init__(self, value: int, propositions: typing.List[typing.Union[str, puan.variable]], variable: typing.Union[str, puan.variable] = None, sign: int = 1):
        self.generated_id = False
        self.value = value
        self.sign = sign
        if not sign in [-1,1]:
            raise Exception(f"`sign` of AtLeast proposition must be either -1 or 1, got: {sign}")

        if sign is None:
            self.sign = 1 if value > 0 else -1

        propositions_list = list(propositions)
        if propositions is None or len(propositions_list) == 0:
            raise Exception("Sub propositions cannot be `None`")

        self.propositions = sorted(
            set(
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
                )
            ),
        )

        if variable is None:
            self.variable = puan.variable(id=AtLeast._id_generator(self.propositions, value, sign))
            self.generated_id = True
        elif type(variable) == str:
            self.variable = puan.variable(id=variable, bounds=(0,1))
        elif type(variable) == puan.variable:
            if variable.bounds.as_tuple() != (0,1):
                raise ValueError(f"variable of a compound proposition cannot have bounds other than (0, 1), got: {variable.bounds}")
            self.variable = variable
        else:
            raise ValueError(f"`variable` must be of type `str` or `puan.variable`, got: {type(variable)}")

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
        return hash(self.variable.id)

    def _id_generator(propositions, value: int, sign: int, prefix: str = "VAR"):
        return prefix + hashlib.sha256(
            str(
                "".join(
                    itertools.chain(
                        map(
                            operator.attrgetter("id"), 
                            filter(lambda x: type(x) == puan.variable, propositions)
                        ),
                        map(
                            operator.attrgetter("variable.id"), 
                            filter(lambda x: type(x) != puan.variable, propositions)
                        )
                    )
                ) + str(value) + str(sign)
            ).encode()
        ).hexdigest()

    @property
    def id(self):
        return self.variable.id

    @property
    def bounds(self):
        return self.variable.bounds

    @property
    def compound_propositions(self) -> iter:
        """
            All propositions of (inheriting) type ``AtLeast``.

            Examples
            --------
                >>> proposition = AtLeast(value=1, propositions=["a", AtLeast(value=2, propositions=list("xy"), variable="B")], variable="A")
                >>> list(proposition.compound_propositions)
                [B: +(x,y)>=2]
        """
        return filter(lambda x: type(x) != puan.variable, self.propositions)

    @property
    def atomic_propositions(self) -> iter:
        """
            All propositions of type ``puan.variable``.

            Examples
            --------
                >>> proposition = AtLeast(value=1, propositions=["a", AtLeast(value=2, propositions=list("xy"), variable="B")], variable="A")
                >>> list(proposition.atomic_propositions)
                [variable(id='a', bounds=Bounds(lower=0, upper=1))]
        """
        return filter(lambda x: type(x) == puan.variable, self.propositions)

    @property
    def variables(self) -> list:

        """
            All (including in sub propositions) variables in this proposition.

            Returns
            -------
                out : typing.List[puan.variable]
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

    def flatten(self) -> list:

        """
            Returns all its propositions and their sub propositions as a unique list of propositions.

            Examples
            --------
                >>> proposition = AtLeast(1, [AtLeast(1, ["a", "b"], "B"), AtLeast(1, ["c", "d"], "C"), "e"], "A")
                >>> proposition.flatten()
                [A: +(B,C,e)>=1, B: +(a,b)>=1, C: +(c,d)>=1, variable(id='a', bounds=Bounds(lower=0, upper=1)), variable(id='b', bounds=Bounds(lower=0, upper=1)), variable(id='c', bounds=Bounds(lower=0, upper=1)), variable(id='d', bounds=Bounds(lower=0, upper=1)), variable(id='e', bounds=Bounds(lower=0, upper=1))]
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

    def to_polyhedron(self, active: bool = False) -> pnd.ge_polyhedron:

        """
            Converts into a polyhedron.

            Properties
            ----------
                active : bool = False
                    If true, then the top node id will be assumed to be true.

            Returns
            -------
                out : pnd.ge_polyhedron
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
        lineqs = pst.TheoryPy(
            list(
                map(
                    lambda x: pst.StatementPy(
                        variable_id_map[x.id][0],
                        variable_id_map[x.id][1].bounds.as_tuple(),
                        pst.AtLeastPy(
                            list(
                                map(
                                    lambda y: variable_id_map[y.id][0], 
                                    x.propositions
                                )
                            ),
                            bias=-1*x.value,
                        ) if type(x) != puan.variable else None,
                    ),
                    flatten_dict.values()
                )
            )
        ).to_lineqs()
        M = np.zeros((len(lineqs), 1+len(variable_id_map)))
        for i, lineq in enumerate(lineqs):
            M[i, list(map(lambda x: x+1, lineq.indices))] = lineq.coeffs
            M[i, 0] = -1*lineq.bias

        polyhedron = pnd.ge_polyhedron(
            M, 
            variables=[puan.variable.support_vector_variable()]+list(
                itertools.chain(
                    map(
                        lambda x: x[1].variable if hasattr(x[1], "variable") else x[1], 
                        variable_id_map.values()
                    )
                )
            ),
            index=list(
                map(
                    lambda x: x[1].variable,
                    filter(
                        lambda x: type(x[1]) != puan.variable, 
                        variable_id_map.values()
                    )
                )
            )
        )

        # assume top node for now until
        # this is default in puan-rspy
        if self.variable in polyhedron.variables and active:

            polyhedron = polyhedron.reduce_columns(
                polyhedron.A.construct(
                    *{self.variable.id: 1}.items(), 
                    default_value=np.nan,
                    dtype=float,
                )
            )

        return polyhedron

    def negate(self) -> "AtLeast":

        """
            Negates proposition.

            Examples
            --------
                >>> AtLeast(2, ["x","y","z"], variable="A").negate()
                A: -(x,y,z)>=-1

                >>> AtLeast(3, ["x","y","z"], variable="A").negate()
                A: -(x,y,z)>=-2

                >>> AtLeast(-1, ["x","y","z"], variable="A", sign=-1).negate()
                A: +(x,y,z)>=2

            Returns
            -------
                out : AtLeast
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

    def diff(self, other) -> list:

        """
            Diff method is part of the model versioning methods helping users
            to keep track of model changes. 
            Diff computes difference between this proposition and another proposition.

            See also
            --------
            revert: Reverts a patch change back to it's original.
            patch: Apply's diff-result onto this proposition.

            Returns
            -------
                out : list
        """
        return list(dictdiffer.diff(self.to_dict(), other.to_dict()))

    def patch(self, diff):

        """
            Patch method is part of the model versioning methods helping users to
            keep track of model changes. Patch apply's diff-result onto this proposition.

            See also
            --------
            revert: Reverts a patch change back to it's original.
            diff: Computes difference between this proposition and another proposition.

            Returns
            -------
                out : AtLeast
        """
        return from_dict(dictdiffer.patch(diff, self.to_dict()))

    def revert(self, diff):

        """
            Revert method is part of the model versioning methods helping user to keep track
            of model changes. Revert reverts a patch change back to it's original.

            See also
            --------
            diff: Computes difference between this proposition and another proposition.
            patch: Apply's diff-result onto this proposition.
        """
        return from_dict(dictdiffer.revert(diff, self.to_dict()))

    @property
    def _equation_mm(self) -> tuple:

        """Max min value of equation exclusive bias"""
        can_min_val = sum(map(lambda x: min(x.bounds.as_tuple())*self.sign, self.propositions))
        can_max_val = sum(map(lambda x: max(x.bounds.as_tuple())*self.sign, self.propositions))
        min_val = min(can_min_val, can_max_val)
        max_val = max(can_min_val, can_max_val)
        return (min_val, max_val)

    @property
    def equation_bounds(self) -> tuple:
        """
            The range of values the left hand side of equations on the form :math:`ax + by + cz - value(bias) \ge 0` can take.

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
                out : tuple
        """
        mn, mx = self._equation_mm
        return (mn-self.value, mx-self.value)

    @property
    def is_tautologi(self) -> bool:

        """
            Returns wheather or not this proposition is true, no matter the interpretation of its propositions.

            Notes
            -----
            Sub propositions are not taken into consideration.

            Examples
            --------
                >>> model = AtLeast(1,["x","y"])
                >>> model.is_tautologi
                False

                >>> model = AtMost(1,["x","y"])
                >>> model.is_tautologi
                False

                >>> model = AtMost(3,["x","y","z"])
                >>> model.is_tautologi
                True

                >>> model = AtLeast(0,["x"])
                >>> model.is_tautologi
                True

                >>> model = AtMost(2,["x","y"])
                >>> model.is_tautologi
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

    def evaluate(self, interpretation: typing.List[puan.SolutionVariable]) -> bool:

        """
            Evaluates interpretation on this model. It will evaluate sub propositions
            bottoms-up and propagate results upwards. This means that even though 
            intermediate variables are not set in interpretation, they receive a value
            based on the evaluation of its propositions.

            Examples
            --------
                >>> All(*"xy", variable="A").evaluate([puan.SolutionVariable("x", value=1)])
                False

                >>> All(*"xy", variable="A").evaluate([puan.SolutionVariable("x", value=1), puan.SolutionVariable("y", value=1)])
                True

                >>> AtLeast(propositions=[puan.variable("x", dtype="int")], value=10).evaluate([puan.SolutionVariable("x", value=9)])
                False

                >>> AtLeast(propositions=[puan.variable("x", dtype="int")], value=10).evaluate([puan.SolutionVariable("x", value=10)])
                True
        """

        interpretation_map = dict(
            zip(
                map(
                    operator.attrgetter("id"),
                    interpretation
                ),
                map(
                    operator.attrgetter("value"),
                    interpretation
                ),
            )
        )

        return (
            sum(
                map(
                    lambda x: interpretation_map.get(x.id, 0)*self.sign if not type(x) == bool else x*1,
                    itertools.chain(
                        self.atomic_propositions,
                        map(
                            operator.methodcaller(
                                "evaluate", 
                                interpretation=interpretation,
                            ),
                            self.compound_propositions
                        )
                    )
                )
            )-self.value
        ) >= 0


    def to_short(self) -> tuple:

        """
            `short` is a tuple format with five element types:
            (0) id, (1) sign, (2) propositions, (3) bias (4) bounds.

            Notes
            -----
                to_short does not include sub propositions if any

            Examples
            --------
                >>> All("x","y","z",variable="A").to_short()
                ('A', 1, ['x', 'y', 'z'], -3, [0, 1])

            Returns
            -------
                out : tuple
        """
        return (self.id, self.sign, list(map(operator.attrgetter("id"), self.propositions)), -1*self.value, list(self.bounds.as_tuple()))

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
        )

    def to_dict(self) -> typing.Dict[str, list]:

        """
            Transforms model into a dictionary representation.
            The key is the id of a proposition.
            The value is a list of three elements:

            #. sign of coeffs (e.g. sign=1 means :math:`a+b+c`, sign=-1 means :math:`-a-b-c`)
            #. sub propositions / variables (e.g. a,b,c)
            #. value of support vector (e.g. 3 as in :math:`a+b+c \ge 3`)

            Examples
            --------
                >>> All(All("a","b", variable="B"), Any("c","d", variable="C"), variable="A").to_dict()
                {'A': (1, ['B', 'C'], -2, [0, 1]), 'B': (1, ['a', 'b'], -2, [0, 1]), 'a': (1, [], 0, (0, 1)), 'b': (1, [], 0, (0, 1)), 'C': (1, ['c', 'd'], -1, [0, 1]), 'c': (1, [], 0, (0, 1)), 'd': (1, [], 0, (0, 1))}
            
            Returns
            -------
                out : typing.Dict[str, list]
        """

        t = self.to_short()
        return {
            **{
                t[0]: t[1:],
            },
            **functools.reduce(lambda x,y: dict(x,**y), map(operator.methodcaller("to_dict"), self.propositions), {})
        }

    def to_json(self) -> dict:

        """
            Returns proposition as a readable json.

            Returns
            -------
                out : dict
        """
        return {
            'id': self.id,
            'type': self.__class__.__name__,
            'propositions': list(
                map(
                    operator.methodcaller("to_json"),
                    self.propositions
                )
            ),
            'value': self.value
        }

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
    def from_json(data: dict, class_map: list) -> "AtLeast":

        """
            Convert from json data to a proposition.

            Properties
            ----------
                class_map : list
                    A list of classes implementing the ``puan.StatementInterface`` protocol.
                    They'll be mapped from `type` -attribute in the json data.

            Notes
            -----
            Propositions within data not having `type` -attribute will be considered a ``puan.variable``.

            Examples
            --------
                >>> AtLeast.from_json({"type": "AtLeast", "propositions": [{"id":"x"},{"id":"y"},{"id":"z"}], "id": "A"}, [puan.variable, AtLeast])
                A: +(x,y,z)>=1

            Returns
            -------
                out : AtLeast
        """
        propositions = data.get('propositions', [])
        return AtLeast(
            value=data.get('value', 1),
            propositions=list(map(functools.partial(from_json, class_map=class_map), propositions)),
            variable=data.get('id', None)
        )

    @staticmethod
    def from_short(short: tuple) -> "AtLeast":

        """
            From short data format into an ``AtLeast`` proposition.
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
                out : AtLeast
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

class AtMost(AtLeast):

    """
        ``AtMost`` proposition is an at least expression with negative coefficients and positive bias 
        (e.g. `:math:-x-y-z+1 \ge 0`). Sub propositions may take on any value given by their equation bounds

        Notes
        -----
            - Propositions may be of type ``str``, ``puan.variable`` or ``AtLeast`` (or other inheriting AtLeast) 
            - Propositions list cannot be empty.

        Examples
        --------
        Meaning at most one of x, y and z.
            >>> AtMost(1, list("xyz"), variable="A")
            A: -(x,y,z)>=-1
    """
    
    def __init__(self, value: int, propositions: typing.List[typing.Union[str, puan.variable]], variable: typing.Union[str, puan.variable] = None):
        super().__init__(value=-1*value, propositions=propositions, variable=variable, sign=-1)

    @staticmethod
    def from_json(data: dict, class_map) -> "AtMost":
        """
            Convert from json data to a proposition.

            Returns
            -------
                out : AtMost
        """
        propositions = data.get('propositions', [])
        return AtMost(
            value=data.get('value', 1),
            propositions=list(map(functools.partial(from_json, class_map=class_map), propositions)),
            variable=data.get('id', None)
        )

    def to_json(self) -> dict:

        """
            Returns proposition as a readable json.

            Returns
            -------
                out : dict
        """
        return {
            'id': self.id,
            'type': self.__class__.__name__,
            'propositions': list(
                map(
                    operator.methodcaller("to_json"),
                    self.propositions
                )
            ),
            'value': -1*self.value
        }

class All(AtLeast):

    """
        ``All`` proposition means that all of its propositions must be true, otherwise the proposition
        is false.

        Notes
        -----
            - Propositions may be of type ``str``, ``puan.variable`` or ``AtLeast`` (or other inheriting AtLeast) 
            - Propositions list cannot be empty.

        Examples
        --------
        Meaning at least one of x, y and z.
            >>> All(*"xyz", variable="A")
            A: +(x,y,z)>=3
    """
    
    def __init__(self, *propositions, variable: typing.Union[str, puan.variable] = None):
        super().__init__(value=len(set(propositions)), propositions=propositions, variable=variable)

    @staticmethod
    def from_json(data: dict, class_map) -> "All":
        """
            Convert from json data to a proposition.

            Returns
            -------
                out : All
        """
        propositions = data.get('propositions', [])
        return All(
            *map(functools.partial(from_json, class_map=class_map), propositions),
            variable=data.get('id', None)
        )

    @staticmethod
    def from_list(propositions: list, variable: typing.Union[str, puan.variable] = None):
        return All(*propositions, variable=variable)

    def to_json(self) -> dict:

        """
            Returns proposition as a readable json.

            Returns
            -------
                out : dict
        """
        return {
            'id': self.id,
            'type': self.__class__.__name__,
            'propositions': list(
                map(
                    operator.methodcaller("to_json"),
                    self.propositions
                )
            ),
        }

class Any(AtLeast):

    """
        ``Any`` proposition means that at least 1 of its propositions must be true, otherwise the proposition
        is false.

        Notes
        -----
            - Propositions may be of type ``str``, ``puan.variable`` or ``AtLeast`` (or other inheriting AtLeast) 
            - Propositions list cannot be empty.

        Examples
        --------
        Meaning at least one of x, y and z.
            >>> All(*"xyz", variable="A")
            A: +(x,y,z)>=3
    """

    def __init__(self, *propositions, variable: typing.Union[str, puan.variable] = None):
        super().__init__(value=1, propositions=propositions, variable=variable)

    @staticmethod
    def from_json(data: dict, class_map) -> "Any":
        """
            Convert from json data to a proposition.

            Returns
            -------
                out : Any
        """
        propositions = data.get('propositions', [])
        return Any(
            *map(functools.partial(from_json, class_map=class_map), propositions),
            variable=data.get('id', None)
        )

    @staticmethod
    def from_list(propositions: list, variable: typing.Union[str, puan.variable] = None):
        return Any(*propositions, variable=variable)

    def to_json(self) -> dict:

        """
            Returns proposition as a readable json.

            Returns
            -------
                out : dict
        """
        return {
            'id': self.id,
            'type': self.__class__.__name__,
            'propositions': list(
                map(
                    operator.methodcaller("to_json"),
                    self.propositions
                )
            ),
        }

class Imply(Any):

    """
        ``Imply`` proposition consists of two sub propositions: condition and consequence. An implication
        proposition says that if the condition is true then the consequence must be true. Otherwise the proposition is false.

        Notes
        -----
            - Propositions may be of type ``str``, ``puan.variable`` or ``AtLeast`` (or other inheriting AtLeast) 
            - Condition and consequence must be set.

        Examples
        --------
        Meaning at least one of x, y and z.
            >>> Imply(condition=All(*"abc", variable="B"), consequence=Any(*"xyz", variable="C"), variable="A")
            A: +(B,C)>=1
    """

    def __init__(self, condition, consequence, variable: typing.Union[str, puan.variable] = None):
        if type(condition) in [str, puan.variable]:
            condition = All(condition)
        self.condition = condition.negate()
        self.consequence = consequence
        super().__init__(self.condition, self.consequence, variable=variable)

    @staticmethod
    def from_json(data: dict, class_map) -> "Imply":
        """
            Convert from json data to a proposition.

            Raises
            ------
                Exception
                    If no `consequence` key in `data`.

            Returns
            -------
                out : Imply
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

    def to_json(self) -> dict:

        """
            Returns proposition as a readable json.

            Returns
            -------
                out : dict
        """
        return {
            'id': self.id,
            'type': self.__class__.__name__,
            'condition': self.condition.negate().to_json(),
            'consequence': self.consequence.to_json(),
        }

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
                someId: +(VAR180540a846781f231b3c1fb0422d95e48e9b9379a5ec6890a0b9a32cb7f66b75,VARb6a05d7d91efc84e49117524cffa01cba8dcb1f14479be025342b909c9ab0cc2)>=1
            
            Returns
            -------
                out : Imply
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
        ``Xor`` proposition is true when exactly one of its propositions is true, e.g. x+y+z = 1

        Notes
        -----
            - Propositions may be of type str, puan.variable or AtLeast (or other inheriting AtLeast) 
            - Propositions list cannot be empty.
    """

    def __init__(self, *propositions, variable: typing.Union[str, puan.variable] = None):
        super().__init__(
            AtLeast(value=1, propositions=propositions), 
            AtMost(value=1, propositions=propositions), 
            variable=variable,
        )

    @staticmethod
    def from_json(data: dict, class_map) -> "Xor":
        """
            Convert from json data to a proposition.

            Returns
            -------
                out : Xor
        """
        propositions = data.get('propositions', [])
        return Xor(
            *map(functools.partial(from_json, class_map=class_map), propositions),
            variable=data.get('id', None)
        )

    @staticmethod
    def from_list(propositions: list, variable: typing.Union[str, puan.variable] = None):
        return Xor(*propositions, variable=variable)

    def to_json(self) -> dict:

        """
            Returns proposition as a readable json.

            Returns
            -------
                out : dict
        """
        return {
            'id': self.id,
            'type': self.__class__.__name__,
            'propositions': list(
                map(
                    operator.methodcaller("to_json"),
                    self.propositions[0].propositions
                )
            ) if len(self.propositions) > 0 else [],
        }

class Not():

    """
        ``Not`` proposition negates the input proposition. 

        Notes
        -----
            - Proposition may be of type ``str``, ``puan.variable`` or ``AtLeast`` (or other inheriting AtLeast) 
    """

    def __new__(self, proposition):
        return (All(proposition) if type(proposition) in [str, puan.variable] else proposition).negate()

    @staticmethod
    def from_json(data: dict, class_map) -> "AtLeast":
        """
            Convert from json data to a proposition.

            Raises
            ------
                Exception
                    If no `proposition` key in `data`.

            Returns
            -------
                out : AtLeast
        """
        if not 'proposition' in data:
            raise Exception("type `Not` expects field `proposition` to have a proposition set")

        return Not(
            from_json(data['proposition'], class_map=class_map),
        )

class XNor():

    """
        ``XNor`` proposition is a negated ``Xor``. I.e. only "exactly one" -configurations will be false. 

        Notes
        -----
            - Proposition may be of type ``str``, ``puan.variable`` or ``AtLeast`` (or other inheriting AtLeast) 
    """

    def __new__(self, *propositions, variable: typing.Union[puan.variable, str] = None):
        return Not(
            Xor(
                *propositions,
                variable=variable
            )
        )

    @staticmethod
    def from_json(data: dict, class_map) -> "XNor":
        """
            Convert from json data to a proposition.

            Returns
            -------
                out : AtLeast
        """
        propositions = data.get('propositions', [])
        return XNor(
            *map(functools.partial(from_json, class_map=class_map), propositions),
            variable=data.get('id', None)
        )

    @staticmethod
    def from_list(propositions: list, variable: typing.Union[str, puan.variable] = None):
        return XNor(*propositions, variable=variable)

def from_json(data: dict, class_map: list = [puan.variable,AtLeast,AtMost,All,Any,Xor,Not,XNor,Imply]) -> typing.Any:

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
            out : dict
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

def from_dict(d: dict, id: str = None) -> AtLeast:

    """
        Transform from dictionary to a compound proposition.
        Values data type is following:
        [int, [str...], int] where 0 is the sign value,
        1 contains the variable names and 2 is the support vector value.
        
        Examples
        --------
            >>> from_dict({'a': [1, ['b','c'], 1, [0,1]], 'b': [1, ['x','y'], 1, [0,1]], 'c': [1, ['p','q'], 1, [0,1]]})
            a: +(b,c)>=-1

        Raises
        ------
            Exception
                | If there exists no topological sort order or a circular dependency exists.
                | If there are more than one top node.
        
        Returns
        -------
            out : AtLeast
    """
    d_conv = dict(
        zip(
            d.keys(),
            map(
                AtLeast.from_short,
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

    # Find top sort order
    try:
        sort_order = list(
            toposort(
                dict(
                    zip(
                        d.keys(),
                        map(
                            lambda x: set(x[1]),
                            d.values()
                        )
                    )
                )
            )
        )
    except Exception as e:
        raise Exception(f"could not create topological sort order from dict: {e}")

    if not len(sort_order[-1]) == 1:
        raise Exception(f"dict has multiple top nodes ({sort_order[-1]}) but exactly one is required")

    for level in sort_order:
        for i in filter(lambda x: x in d_conv, level):
            d_conv[i].propositions = list(
                map(
                    lambda j: d_conv[j] if j in d_conv else j,
                    d_conv[i].propositions
                )
            )

    return d_conv[list(sort_order[-1])[0]]