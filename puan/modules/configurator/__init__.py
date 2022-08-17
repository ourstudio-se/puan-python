import itertools
import maz
import functools
import operator
import numpy as np
import puan
import puan.logic.plog as pg
import puan.ndarray as pnd
import typing
import dataclasses

@dataclasses.dataclass(frozen=True)
class SolutionVariable(puan.variable):

    value: int

    @staticmethod
    def from_variable(variable: puan.variable, value: int) -> "SolutionVariable":
        return SolutionVariable(variable.id, variable.dtype, variable.virtual, value)

class Any(pg.Any):

    """
        Overriding plog.Any proposition with default -attribute indicating which solution
        is default. If proposition is A: a+b+c >= 1 and default is "c", then new proposition is
            A  : A'+c >=1
            A' : a+b  >=1

        and prio is set to c > A'.
    """

    def __init__(self, *propositions, default: str = None, id: str = None):
        self.default = default
        if default is not None:
            inner = pg.Any(
                *filter(lambda x: not x == default, propositions)
            )
            inner.prio = getattr(inner, 'prio', -1)-1 
            super().__init__(
                *filter(lambda x: x == default, propositions),
                inner,
                id=id,
            )
        else:
            super().__init__(*propositions, id=id)

    def to_json(self) -> dict:
        d = super().to_json()
        # currently we only care about first element since no implementation
        # to handle list of defaults (like a prio list). But since future may include list of 
        # defaults, we keep interface as list
        if any(map(operator.attrgetter("virtual"), self.propositions)):
            defaults = list(filter(maz.compose(operator.not_, operator.attrgetter("virtual")), self.propositions))
            if defaults:
                default_prop = defaults[0]
                non_default_prop = next(filter(operator.attrgetter("virtual"), self.propositions))
                d['propositions'] = list(
                    itertools.chain(
                        map(operator.methodcaller("to_json"), non_default_prop.propositions),
                        [default_prop.to_json()]
                    )
                )
                d['default'] = [default_prop.id]

        return d

    @staticmethod
    def from_json(data: dict, class_map: list) -> pg.Proposition:
        
        """"""
        default = data.get("default", [])
        default_item = None if len(default) == 0 else default[0]
        _class_map = dict(zip(map(lambda x: x.__name__, class_map), class_map))
        return Any(
            *map(functools.partial(pg.from_json, class_map=class_map), data.get('propositions', [])),
            default=default_item,
            id=data.get("id", None)
        )

class Xor(pg.All):

    """
        Overriding plog.Xor proposition with default -attribute indicating which solution
        is default. If proposition is A: a+b+c >= 1 and default is "c", then new proposition is
        A  : A'+c >=1
        A' : a+b  >=1
        and prio is set to c > A'.
    """

    def __init__(self, *propositions, default: str = None, id: str = None):
        propositions = list(propositions)
        if default is not None:
            super().__init__(
                Any(*propositions, default=default, id=f"{id}_LB" if not id is None else None),
                pg.AtMost(*propositions, value=1, id=f"{id}_UB" if not id is None else None),
                id=id
            )
        else:
            super().__init__(
                pg.Any(*propositions, id=f"{id}_LB" if not id is None else None),
                pg.AtMost(*propositions, value=1, id=f"{id}_UB" if not id is None else None),
                id=id
            )

    def to_json(self):
        d = super().to_json()
        ls, rs = d['propositions']
        d['propositions'] = rs['propositions']
        d['default'] = ls['default']
        return d

    @staticmethod
    def from_json(data: dict, class_map: list) -> pg.Proposition:

        """"""
        default = data.get("default", [])
        default_item = None if len(default) == 0 else default[0]
        _class_map = dict(zip(map(lambda x: x.__name__, class_map), class_map))
        return Xor(
            *map(functools.partial(pg.from_json, class_map=class_map), data.get('propositions', [])),
            default=default_item,
            id=data.get("id", None)
        )

class StingyConfigurator(pg.All):

    """
        A class for supporting a sequential configurator experience.
        The "stingy" configurator always tries to select the least possible number of selections in a solution, with
        respect to what's been prioritized.
        Whenever a AtLeast -proposition proposes multiple solutions that equal least number
        of selections, then a default may be added to avoid ambivalence.
    """

    def __init__(self, *propositions: typing.List[typing.Union[pg.Proposition, str]], id: str = None):
        super().__init__(*propositions, id=id)

    @property
    @functools.lru_cache
    def polyhedron(self) -> pnd.ge_polyhedron:

        """
            This configurator model's polyhedron (see logic.plog.Proposition.to_polyhedron).
        """
        
        return self.to_polyhedron(True)
    
    @property
    @functools.lru_cache
    def default_prio_vector(self) -> np.ndarray:

        """
            Is the objective vector when nothing is selected.
        """
        return np.array(
            list(
                map(
                    lambda p: getattr(p, "prio", 0 if p.virtual else -1),
                    sorted(
                        self.flatten(),
                        key=maz.compose(
                            self.variables.index, 
                            operator.attrgetter("id")
                        )
                    ),
                ),
            ), 
            dtype=np.int32
        )

    def _vectors_from_prios(self, prios: typing.List[typing.Dict[str, int]]) -> np.ndarray:

        """
            Constructs weight vectors from prioritization list.
        """

        return pnd.integer_ndarray(
            np.array(
                list(
                    map(
                        lambda y: [
                            self.default_prio_vector.tolist(),
                            list(map(lambda x: dict.get(y, x.id, 0), self.polyhedron.A.variables))
                        ],
                        prios
                    )
                )
            )
        ).ndint_compress(method="shadow", axis=0)

    def select(self, *prios: typing.List[typing.Dict[str, int]], solver, include_virtual_vars: bool = False) -> typing.Iterable[typing.List[puan.variable]]:

        """
            Select items to prioritize and receive a solution.

            Parameters
            ----------
                *prios : typing.List[typing.Dict[str, int]]
                    a list of dicts where each entry's value is a prio

                solver : a mixed integer linear programming solver

            Returns
            -------
                out : typing.List[typing.List[puan.variable]]
        """
        return map(
            lambda v: list(
                filter(
                    maz.compose(
                        functools.partial(
                            operator.add,
                            include_virtual_vars
                        ),
                        operator.not_,
                        operator.attrgetter("virtual"),
                    ),
                    itertools.starmap(
                        SolutionVariable.from_variable,
                        zip(self.polyhedron.A.variables[v > 0], v[v > 0].tolist())
                    ),
                )
            ), 
            solver(
                self.polyhedron.A, 
                self.polyhedron.b,
                self.polyhedron.A.integer_variable_indices(),
                self._vectors_from_prios(prios),
            )
        )

    def add(self, proposition: pg.Proposition) -> "StingyConfigurator":

        """
            Add a new proposition to the model.

            Returns
            -------
                out : StingyConfigurator
        """
        return StingyConfigurator(
            *(self.propositions + [proposition]), 
            id=self.id,
        )
    
    def assume(self, fixed: typing.Dict[str, int]) -> tuple:

        """
            Assumes variables in `fixed`-list. Passes prio onwards as well.

            Returns
            -------
                out : tuple(StingyConfigurator, dict)
        """
        # Prepare to put pack prio's
        self_flatten = self.flatten()
        d_flatten = dict(
            zip(
                map(
                    operator.attrgetter("id"), 
                    self_flatten,
                ), 
                self_flatten,
            ),
        )

        assumed_sub, variable_consequence = pg.AtLeast(
            *self.propositions, 
            value=len(self.propositions), 
            id=self._id,
        ).assume(fixed)
        
        # Put back prio into proposition with prio set
        for proposition in filter(lambda x: isinstance(x, pg.Any), assumed_sub.flatten()):
            proposition.__class__ = Any
            proposition.prio = getattr(d_flatten.get(proposition.id, {}), "prio", -1)

        return StingyConfigurator(
            *assumed_sub.propositions,
            id=self._id,
        ), variable_consequence

    def from_json(data: dict) -> "StingyConfigurator":

        """"""
        classes = [
            pg.Proposition,
            pg.AtLeast,
            pg.AtMost,
            pg.All,
            Any,
            Xor,
            pg.Not,
            pg.XNor,
            pg.Imply,
        ]
        return StingyConfigurator(
            *map(
                functools.partial(pg.from_json, class_map=classes),
                data.get('propositions', [])
            ),
            id=data.get('id', None)
        )

    def to_json(self) -> dict:
        d = super().to_json()
        return d