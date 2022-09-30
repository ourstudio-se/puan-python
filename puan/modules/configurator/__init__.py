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

class Any(pg.Any):

    """
        Overriding plog.Any proposition with default -attribute indicating which solution
        is default. If proposition is A: a+b+c >= 1 and default is "c", then new proposition is
            A  : A'+c >=1
            A' : a+b  >=1

        and prio is set to c > A'.
    """

    def __init__(self, *propositions, default: str = None, variable: typing.Union[puan.variable, str] = None):
        self.default = default
        if default is not None:
            inner = pg.Any(
                *filter(lambda x: not x == default, propositions)
            )
            inner.prio = getattr(inner, 'prio', -1)-1 
            super().__init__(
                *filter(lambda x: x == default, propositions),
                inner,
                variable=variable,
            )
        else:
            super().__init__(*propositions, variable=variable)

    def to_json(self) -> dict:
        d = super().to_json()
        # currently we only care about first element since no implementation
        # to handle list of defaults (like a prio list). But since future may include list of 
        # defaults, we keep interface as list
        defaults = list(filter(lambda x: hasattr(x, "prio"), self.propositions))
        if defaults:
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
    def from_json(data: dict, class_map: list) -> pg.AtLeast:
        
        """"""
        default = data.get("default", [])
        default_item = None if len(default) == 0 else default[0]
        _class_map = dict(zip(map(lambda x: x.__name__, class_map), class_map))
        return Any(
            *map(functools.partial(pg.from_json, class_map=class_map), data.get('propositions', [])),
            default=default_item,
            variable=data.get("id", None)
        )

class Xor(pg.Xor):

    """
        Overriding plog.Xor proposition with default -attribute indicating which solution
        is default. If proposition is A: a+b+c >= 1 and default is "c", then new proposition is
        A  : A'+c >=1
        A' : a+b  >=1
        and prio is set to c > A'.
    """

    def __init__(self, *propositions, default: str = None, variable: typing.Union[puan.variable, str] = None):
        super().__init__(*propositions, variable=variable)
        
        # From init, there should be exactly one Any constraint
        if default is not None:
            i, any_proposition = next(filter(lambda x: x[1].value == 1, enumerate(self.propositions)))
            self.propositions[i] = Any(
                *any_proposition.propositions, 
                default=default, 
                variable=any_proposition.variable
            )
            

    def to_json(self):
        d = super().to_json()
        ls, rs = d['propositions']
        d['propositions'] = rs['propositions']
        d['default'] = ls['default']
        return d

    @staticmethod
    def from_json(data: dict, class_map: list) -> pg.AtLeast:

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

    def __init__(self, *propositions: typing.List[typing.Union[pg.AtLeast, str]], id: str = None):
        super().__init__(*propositions, variable=id)

    @property
    @functools.lru_cache
    def polyhedron(self) -> pnd.ge_polyhedron_config:

        """
            This configurator model's polyhedron (see logic.plog.Proposition.to_polyhedron).
        """
        polyhedron = self.to_polyhedron(True)
        return pnd.ge_polyhedron_config(
            polyhedron, 
            default_prio_vector=polyhedron.A.construct(*self.default_prios.items()),
            variables=polyhedron.variables, 
            index=polyhedron.index, 
        )

    @property
    def default_prios(self) -> dict:

        """
            Default prio dictionary, based on defaults in either Xor's or Any's.

            Returns
            -------
                out : dict
        """
        flat = self.flatten()
        return dict(
            zip(
                map(operator.attrgetter("id"), flat),
                map(
                    lambda p: getattr(p, "prio", -1),
                    flat
                )
            )
        )

    @functools.lru_cache
    def leafs(self) -> typing.List[puan.variable]:

        """
            Return propositions that are of type puan.variable.

            Returns
            -------
                out : typing.List[puan.variable]
        """
        flatten = self.flatten()
        return sorted(
            set(
                itertools.chain(
                    filter(
                        lambda x: type(x) == puan.variable,
                        flatten
                    ),
                ),
            )
        )
    

    def select(self, *prios: typing.List[typing.Dict[str, int]], solver, only_leafs: bool = False) -> typing.Iterable[typing.List[puan.variable]]:

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
        res = self.polyhedron.select(
            *prios, 
            solver=solver, 
        )
        if only_leafs:
            leafs = list(
                map(
                    operator.attrgetter("id"),
                    self.leafs()
                )
            )
            res = map(
                lambda config: list(
                    filter(
                        lambda x: x.id in leafs,
                        config
                    )
                ),
                res
            )
        return res

    def add(self, proposition: pg.AtLeast) -> "StingyConfigurator":

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
            puan.variable,
            pg.AtLeast,
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