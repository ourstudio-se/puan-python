import maz
import functools
import operator
import numpy as np
import puan
import puan.logic.plog as pg
import puan.ndarray as pnd
import typing

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
            inner = super.__init__(
                *filter(lambda x: not x == default, propositions),
                value=1
            )
            inner.prio = getattr(inner, 'prio', -1)-1 
            super.__init__(
                *filter(lambda x: x == default, propositions),
                inner,
                value=1,
                id=id,
            )
        else:
            super().__init__(*propositions, id=id)

    def to_json(self) -> dict:
        d = super().to_json()
        d['default'] = self.default
        return d

class Xor(pg.All):

    """
        Overriding plog.Xor proposition with default -attribute indicating which solution
        is default. If proposition is A: a+b+c >= 1 and default is "c", then new proposition is
        A  : A'+c >=1
        A' : a+b  >=1
        and prio is set to c > A'.
    """

    def __init__(self, *propositions, default: str = None, id: str = None):
        if default is not None:
            super().__init__(
                Any(*propositions, default=default, id=f"{id}_LB" if not id is None else None),
                pg.AtMost(*propositions, value=1, id=f"{id}_UB" if not id is None else None),
                id=id
            )
        else:
            super().__init__(*propositions, id=id)

    def to_json(self):
        d = super().to_json()
        ls, rs = d['propositions']
        d['propositions'] = rs['propositions']
        d['default'] = ls['default']
        return d


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

        prios = set([(self.id, getattr(self, 'prio', -1))])
        queue = set(self.propositions)
        while len(queue) > 0:
            prop = queue.pop()
            prios.add((prop.id, getattr(prop, 'prio', -1)))
            queue.update(getattr(prop, 'propositions', []))

        return np.array(list(map(operator.itemgetter(1), sorted(prios, key=operator.itemgetter(0)))), dtype=np.int32)

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
                    lambda vvr: not vvr[0].virtual or include_virtual_vars, 
                    zip(self.polyhedron.A.variables[v > 0], map(int, v[v > 0])),
                )
            ), 
            solver(
                self.polyhedron.A, 
                self.polyhedron.b,
                self.polyhedron.A.integer_variable_indices(),
                self._vectors_from_prios(prios),
                # -np.ones((1, self.polyhedron.A.shape[1]))
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
    
    def assume(self, *fixed: typing.List[str]) -> "StingyConfigurator":

        """
            Assumes variables in `fixed`-list. Passes prio onwards as well.

            Returns
            -------
                out : StingyConfigurator
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

        assumed_sub = pg.AtLeast(
            *self.propositions, 
            value=len(self.propositions), 
            id=self._id,
        ).assume(*fixed)
        
        # Put back prio into proposition with prio set
        list(
            map(
                lambda fp: object.__setattr__(fp, "prio", getattr(d_flatten.get(fp.id, {}), "prio", -1)),
                assumed_sub.flatten()
            )
        )

        return StingyConfigurator(
            *assumed_sub.propositions,
            id=self._id,
        )

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