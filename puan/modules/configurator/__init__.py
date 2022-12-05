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
        Overriding :class:`plog.Any<puan.logic.plog.Any>` proposition with default attribute indicating which solution
        is default. If proposition is :math:`A`: :math:`a+b+c \\ge 1` and default is :math:`c`, then new proposition is

            | :math:`A`  : :math:`A'+c \\ge 1`
            | :math:`A'` : :math:`a+b  \\ge 1`

        and prio is set to :math:`c > A'`.

        Parameters
        ----------
        *propositions : an iterable of :class:`puan.Proposition` instances or ``str``
        default : marks which of all sub propositions are default 
        variable : variable connected to this proposition

        Methods
        -------
        to_json
        from_json
        from_list
    """

    def __init__(self, *propositions, default: typing.List[typing.Union[puan.Proposition, str]] = None, variable: typing.Union[puan.variable, str] = None):
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

    def to_json(self) -> typing.Dict[str, typing.Any]:
        # currently we only care about first element since no implementation
        # to handle list of defaults (like a prio list). But since future may include list of 
        # defaults, we keep interface as list
        """
            Converts :class:`Any` to JSON format.

            Returns
            -------
                out : Dict[str, Any]
        """
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
    def from_json(data: typing.Dict[str, typing.Any], class_map: typing.List["Any"]) -> "Any":
        
        """
            Creates :class:`Any` from JSON format.

            Parameters
            ----------
            data : Dict[str, Any]
                JSON data
            class_map : List[Any]
                List of classes which maps to type

            Returns
            -------
                out : :class:`Any`
        """
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
        Overriding :class:`plog.Xor<puan.logic.plog.Xor>` proposition with default attribute indicating which solution
        is default. If proposition is :math:`A`: :math:`a+b+c \\ge 1` and default is :math:`c`, then new proposition is
        
            | :math:`A`  : :math:`A'+c \\ge 1`
            | :math:`A'` : :math:`a+b \\ge 1`
            
        and prio is set to :math:`c > A'`.

        Parameters
        ----------
        *propositions : an iterable of :class:`puan.Proposition` instances or ``str``
        default : marks which of all sub propositions are default 
        variable : variable connected to this proposition

        Methods
        -------
        to_json
        from_json
        from_list
    """

    def __init__(self, *propositions, default: typing.List[typing.Union[str, puan.Proposition]] = None, variable: typing.Union[puan.variable, str] = None):
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
        return Xor(
            *map(functools.partial(pg.from_json, class_map=class_map), data.get('propositions', [])),
            default= None if len(default) == 0 else list(map(lambda x: puan.variable.from_json(x, [puan.variable]), default)),
            variable=data.get("id", None)
        )

    @staticmethod
    def from_list(propositions: list, variable: typing.Union[str, puan.variable] = None, default: typing.List[puan.variable] = []) -> "Xor":
        return Xor(*propositions, variable=variable, default=default)

class StingyConfigurator(pg.All):

    """
        A class for supporting a sequential configurator experience.
        The "stingy" configurator always tries to select the least possible number of selections in a solution, with
        respect to what's been prioritized.
        Whenever an AtLeast proposition proposes multiple solutions that equal least number
        of selections, then a default may be added to avoid ambivalence.

        Parameters
        ----------
        *propositions : an iterable of :class:`puan.Proposition` instances or ``str``
        id : id connected to this proposition

        Methods 
        -------
        ge_polyhedron
        default_prios
        leafs
        select 
        add
        from_json
        to_json
    """

    def __init__(self, *propositions: typing.List[typing.Union[puan.Proposition, str]], id: str = None):
        super().__init__(*propositions, variable=id)

    @property
    @functools.lru_cache
    def ge_polyhedron(self) -> pnd.ge_polyhedron_config:

        """
            This configurator model's polyhedron (see :meth:`plog.AtLeast.to_ge_polyhedron<puan.logic.plog.AtLeast.to_ge_polyhedron>`).

            Returns
            -------
                out : :class:`puan.ndarray.ge_polyhedron_config`
        """
        ge_polyhedron = self.to_ge_polyhedron(True)
        return pnd.ge_polyhedron_config(
            ge_polyhedron, 
            default_prio_vector=ge_polyhedron.A.construct(self.default_prios),
            variables=ge_polyhedron.variables, 
            index=ge_polyhedron.index, 
        )

    @property
    def default_prios(self) -> typing.Dict[str, int]:

        """
            Default prio dictionary, based on defaults in either Xor's or Any's.

            Returns
            -------
                out : Dict[str, int]
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
            Return propositions that are of type :class:`variable<puan.variable>`.

            Returns
            -------
                out : List[:class:`variable<puan.variable>`]
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
    

    def select(self, *prios: typing.List[typing.Dict[str, int]], solver: typing.Callable = None, only_leafs: bool = False) -> itertools.starmap:

        """
            Select items to prioritize and receive a solution.

            Parameters
            ----------
                *prios : List[Dict[str, int]]
                    a list of dicts where each entry's value is a prio

                solver : Callable[[:class:`puan.ndarray.ge_polyhedron`, Dict[str, int]], List[(:class:`np.ndarray`, int, int)]] = None
                    A mixed integer linear programming solver function. Check :meth:`plog.AtLeast.solve <puan.logic.plog.AtLeast.solve>` function for more information on the solver interface.

                only_leafs : ``bool``
                    Controls if only leafs should be returned in the output, see :meth:`leafs`. Default ``False``. 

            Returns
            -------
                out : :class:`itertools.starmap`
        """
        res = self.ge_polyhedron.select(
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
            res = itertools.starmap(
                lambda config,ov,sc: dict(
                    filter(
                        lambda x: x[0] in leafs,
                        config.items()
                    )
                ),
                res
            )
        return res

    def add(self, proposition: pg.AtLeast) -> "StingyConfigurator":

        """
            Add a new proposition to the model.

            Parameters
            ----------
                proposition : pg.AtLeast
                    Proposition to be added

            Raises
            ------
                Exception
                    If already proposition with same id exists among this proposition's sub propositions.


            Returns
            -------
                out : :class:`StingyConfigurator`
        """
        if proposition.id in map(operator.attrgetter("id"), self.propositions):
            raise Exception(f"proposition with id `{proposition.id}` already exist among this proposition's sub propositions")

        return StingyConfigurator(
            *(self.propositions + [proposition]), 
            id=self.id,
        )

    def from_json(data: dict) -> "StingyConfigurator":

        """
            Creates a :class:`StingyConfigurator` from the JSON format.

            Returns
            -------
                out : :class:`StingyConfigurator`
        """
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

    def to_json(self) -> typing.Dict[str, typing.Any]:
        """
            Converts a :class:`StingyConfigurator` to the JSON format
        """
        d = super().to_json()
        return d