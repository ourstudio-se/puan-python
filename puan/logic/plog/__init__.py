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
from dataclasses import dataclass

class AtLeast(puan.StatementInterface):
    
    def __init__(self, value: int, propositions: list = None, variable: typing.Union[str, puan.variable] = None, sign: int = 1):
        self.value = value
        self.sign = sign
        if not sign in [-1,1]:
            raise Exception(f"`sign` of AtLeast proposition must be either -1 or 1, got: {sign}")

        if sign is None:
            self.sign = 1 if value > 0 else -1

        self.propositions = propositions
        if self.propositions is not None:
            self.propositions = list(
                itertools.chain(
                    filter(
                        lambda x: type(x) != str, 
                        self.propositions
                    ),
                    map(
                        puan.variable,
                        filter(
                            lambda x: type(x) == str,
                            self.propositions
                        )
                    )
                )
            )

        if variable is None and propositions is None:
            raise Exception("Proposition must have variable and/or sub propositions, not none")
        elif variable is None:
            self.variable = puan.variable(id=AtLeast._id_generator(self.propositions, value))
        elif type(variable) == str:
            self.variable = puan.variable(id=variable, bounds=(0,1))
        else:
            self.variable = variable

    def __repr__(self) -> str:
        atoms = sorted(
            map(lambda x: x.id, self.propositions)
        )
        return f"{self.variable.id}: {'+' if self.sign > 0 else '-'}({','.join(atoms)})>={self.value}"

    def __lt__(self, other):
        return self.id < other.id

    def __eq__(self, other):
        return (self.id == other.id) & (self.equation_bounds == other.equation_bounds) & (self.value == other.value)

    def __hash__(self):
        return hash(self.variable.id)

    def _id_generator(propositions, value: int, prefix: str = "VAR"):
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
                ) + str(value)
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
        return filter(lambda x: type(x) != puan.variable, self.propositions)

    @property
    def atomic_propositions(self) -> iter:
        return filter(lambda x: type(x) == puan.variable, self.propositions)

    @property
    def variables(self) -> list:

        """
            All (including in sub propositions) variables in this proposition.

            Returns
            -------
                out : typing.List[puan.variable]
        """
        flat = self.flatten()
        return sorted(
            itertools.chain(
                map(
                    operator.attrgetter('variable'),
                    filter(
                        lambda x: type(x) != puan.variable
                    )
                ),
                filter(
                    lambda x: type(x) == puan.variable
                )
            )
        )

    def flatten(self) -> list:
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
        variable_id_map_rev = dict(variable_id_map.values())
        lineqs = list(
            filter(
                lambda x: len(variable_id_map_rev[x[0]].propositions) > 0, 
                enumerate(
                    pst.TheoryPy(
                        list(
                            map(
                                lambda x: pst.StatementPy(
                                    variable_id_map[x.id][0],
                                    variable_id_map[x.id][1].bounds,
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
                )
            )
        )
        M = np.zeros((len(lineqs), 1+len(variable_id_map)))
        for i, lineq in lineqs:
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
            variable=self.variable,
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
                out : Proposition
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
        can_min_val = sum(map(lambda x: min(x.bounds)*self.sign, self.propositions))
        can_max_val = sum(map(lambda x: max(x.bounds)*self.sign, self.propositions))
        min_val = min(can_min_val, can_max_val)
        max_val = max(can_min_val, can_max_val)
        return (min_val, max_val)

    @property
    def equation_bounds(self) -> tuple:
        """
            The max and min value this equation can return.
            ax + by + cz - value (bias)

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
        return self.equation_bounds[1] < self.value

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
        return (self.id, self.sign, list(map(operator.attrgetter("id"), self.propositions)), -1*self.value, list(self.bounds))

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
            1. sign of coeffs (e.g. sign=1 means a+b+c, sign=-1 means -a-b-c)
            2. sub propositions / variables (e.g. a,b,c)
            3. value of support vector (e.g. 3 as in a+b+c>=3)
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
    def from_json(data: dict, class_map) -> "AtLeast":

        """
            Convert from json data to a Proposition.

            Returns
            -------
                out : Proposition
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
            From short data format into AtLeast proposition.

            Returns
            -------
                out : AtLeast
        """
        try:
            _id, sign, props, bias, bounds = short
        except Exception as e:
            raise Exception(f"tried to convert short into AtLeast propositions but failed due to: {e}")

        return AtLeast(
            value=-1*bias,
            propositions=props,
            variable=puan.variable(_id, bounds),
            sign=sign,
        )

class AtMost(AtLeast):
    
    def __init__(self, value: int, propositions: list = None, variable: typing.Union[str, puan.variable] = None):
        super().__init__(value=-1*value, propositions=propositions, variable=variable, sign=-1)

    @staticmethod
    def from_json(data: dict, class_map) -> "AtMost":
        """
            Convert from json data to a Proposition.

            Returns
            -------
                out : Proposition
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

    def __init__(self, *propositions, variable: typing.Union[str, puan.variable] = None):
        super().__init__(value=len(propositions), propositions=propositions, variable=variable)

    @staticmethod
    def from_json(data: dict, class_map) -> "All":
        """
            Convert from json data to a Proposition.

            Returns
            -------
                out : Proposition
        """
        propositions = data.get('propositions', [])
        return All(
            *map(functools.partial(from_json, class_map=class_map), propositions),
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
        }

class Any(AtLeast):

    def __init__(self, *propositions, variable: typing.Union[str, puan.variable] = None):
        super().__init__(value=1, propositions=propositions, variable=variable)

    @staticmethod
    def from_json(data: dict, class_map) -> "Any":
        """
            Convert from json data to a Proposition.

            Returns
            -------
                out : Proposition
        """
        propositions = data.get('propositions', [])
        return Any(
            *map(functools.partial(from_json, class_map=class_map), propositions),
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
        }

class Imply(Any):

    def __init__(self, condition, consequence, variable: typing.Union[str, puan.variable] = None):
        super().__init__(condition, consequence, variable=variable)
        self.propositions[0] = (All(self.propositions[0]) if type(self.propositions[0]) == puan.variable else self.propositions[0]).negate()

    @staticmethod
    def from_json(data: dict, class_map) -> "Imply":
        """
            Convert from json data to a Proposition.

            Returns
            -------
                out : Proposition
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
            'condition': self.propositions[0].negate().to_json(),
            'consequence': self.propositions[1].to_json(),
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
                someId: +(VAR724b37bcf910ce0504f0b4ae3182d4b5c98e65aa937eeb95985d426c8ac76731,VAR86fca5ee438a294cecf9003ee4f8c73dcf99f07273ba6d609ed49abe9e2d3d2d)>=1
            
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

    def __init__(self, *propositions, variable: typing.Union[str, puan.variable] = None):
        super().__init__(
            AtLeast(value=1, propositions=propositions), 
            AtMost(value=1, propositions=propositions), 
            variable=variable,
        )

    @staticmethod
    def from_json(data: dict, class_map) -> "Xor":
        """
            Convert from json data to a Proposition.

            Returns
            -------
                out : Proposition
        """
        propositions = data.get('propositions', [])
        return Xor(
            *map(functools.partial(from_json, class_map=class_map), propositions),
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
                    self.propositions[0].propositions
                )
            ) if len(self.propositions) > 0 else [],
        }

class Not():

    def __new__(self, proposition):
        return (All(proposition) if type(proposition) == str else proposition).negate()

    @staticmethod
    def from_json(data: dict, class_map) -> "Xor":
        """
            Convert from json data to a Proposition.

            Returns
            -------
                out : Proposition
        """
        if not 'proposition' in data:
            raise Exception("type `Not` expects field `proposition` to have a proposition set")

        return Not(
            from_json(data['proposition'], class_map=class_map),
        )

class XNor():

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
            Convert from json data to a Proposition.

            Returns
            -------
                out : Proposition
        """
        propositions = data.get('propositions', [])
        return XNor(
            *map(functools.partial(from_json, class_map=class_map), propositions),
            variable=data.get('id', None)
        )

def from_json(data: dict, class_map: list = [puan.variable,AtLeast,AtMost,All,Any,Xor,Not,XNor,Imply]) -> typing.Any:

    """
        Convert from json data to a Proposition.

        Returns
        -------
            out : Proposition
    """
    _class_map = dict(zip(map(lambda x: x.__name__, class_map), class_map))
    if 'type' not in data or data['type'] in ["Proposition", "Variable"]:
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