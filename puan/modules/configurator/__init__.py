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
        Overriding ``plog.Any`` proposition with default -attribute indicating which solution
        is default. If proposition is A: a+b+c >= 1 and default is "c", then new proposition is

            | A  : A'+c >=1
            | A' : a+b  >=1

        and prio is set to c > A'.
    """

    def __init__(self, *propositions, default: typing.List[typing.Union[puan.variable, str]] = None, variable: typing.Union[puan.variable, str] = None):
        self.default = list(map(lambda x: puan.variable(x) if type(x) == str else x, default if default is not None else []))
        if len(self.default) > 0 and self.default is not None and len(propositions) > 1:
            _default = self.default[0].id
            complement = list(filter(lambda x: not x == _default, propositions))
            if len(complement) == len(propositions) or len(complement) == 0:
                super().__init__(*propositions, variable=variable)
            else:
                inner = pg.Any(*complement)
                inner.prio = getattr(inner, 'prio', -1)-1 
                super().__init__(
                    *filter(lambda x: x == _default, propositions),
                    inner,
                    variable=variable,
                )
        else:
            super().__init__(*propositions, variable=variable)

    def to_json(self) -> dict:
        # currently we only care about first element since no implementation
        # to handle list of defaults (like a prio list). But since future may include list of 
        # defaults, we keep interface as list
        if len(self.propositions) == 2 and any(map(lambda x: hasattr(x, 'prio'), self.propositions)):
            d = {
                "id": self.id,
                "type": "Any",
                "propositions": list(
                    itertools.chain(
                        map(
                            operator.methodcaller("to_json"), 
                            filter(
                                lambda x: not hasattr(x, 'prio'),
                                self.propositions,
                            )
                        ),
                        map(
                            operator.methodcaller('to_json'),
                            next(
                                filter(
                                    lambda x: hasattr(x, 'prio'),
                                    self.propositions,
                                )
                            ).propositions
                        ),
                    ),
                ),
            }
        else:
            d = super().to_json()

        d['default'] = list(map(lambda x: x.to_json(), self.default))

        return d

    @staticmethod
    def from_json(data: dict, class_map: list) -> "Any":
        
        """"""
        default = data.get("default", [])
        if default:
            default_item = None if len(default) == 0 else list(map(lambda x: puan.variable.from_json(x, [puan.variable]), default))
            _class_map = dict(zip(map(lambda x: x.__name__, class_map), class_map))
            return Any(
                *map(functools.partial(pg.from_json, class_map=class_map), data.get('propositions', [])),
                default=default_item,
                variable=data.get("id", None)
            )
        else:
            return pg.Any.from_json(data, class_map)

    
    @staticmethod
    def from_list(propositions: list, variable: typing.Union[str, puan.variable] = None, default: typing.List[puan.variable] = []) -> "Any":
        return Any(*propositions, variable=variable, default=default)

class Xor(pg.Xor):

    """
        Overriding ``plog.Xor`` proposition with default -attribute indicating which solution
        is default. If proposition is A: a+b+c >= 1 and default is "c", then new proposition is
        
            | A  : A'+c >=1
            | A' : a+b  >=1
            
        and prio is set to c > A'.
    """

    def __init__(self, *propositions, default: typing.List[typing.Union[str, puan.variable]] = None, variable: typing.Union[puan.variable, str] = None):
        super().__init__(*propositions, variable=variable)
        
        # From init, there should be exactly one Any constraint
        self.default = list(map(lambda x: puan.variable(x) if type(x) == str else x, default if default is not None else []))
        if self.default:
            i, any_proposition = next(filter(lambda x: x[1].value == 1, enumerate(self.propositions)))
            self.propositions[i] = Any(
                *any_proposition.propositions, 
                default=default, 
                variable=any_proposition.variable
            )
            

    def to_json(self):
        if self.default:
            d = {
                "id": self.id,
                "type": "Xor",
                "propositions": list(
                    map(
                        operator.methodcaller('to_json'),
                        next(filter(lambda x: type(x) == pg.AtMost, self.propositions)).propositions
                    )
                ),
                "default": list(map(lambda x: x.to_json(), self.default))
            }
        else:
            d = super().to_json()

        return d

    @staticmethod
    def from_json(data: dict, class_map: list) -> "Xor":

        """"""
        default = data.get("default", [])
        if default:
            default_item = None if len(default) == 0 else list(map(lambda x: puan.variable.from_json(x, [puan.variable]), default))
            _class_map = dict(zip(map(lambda x: x.__name__, class_map), class_map))
            return Xor(
                *map(functools.partial(pg.from_json, class_map=class_map), data.get('propositions', [])),
                default=default_item,
                variable=data.get("id", None)
            )
        else:
            return pg.Xor.from_json(data, class_map)

    @staticmethod
    def from_list(propositions: list, variable: typing.Union[str, puan.variable] = None, default: typing.List[puan.variable] = []) -> "Xor":
        return Xor(*propositions, variable=variable, default=default)

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
            This configurator model's polyhedron (see logic.plog.AtLeast.to_polyhedron).
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