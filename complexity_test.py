import functools
import operator
import itertools

import puan
import puan.logic.plog as pg
import puan.modules.configurator as cc
import itertools
import more_itertools
import pandas as pd
# import seaborn as sn
import puan_solvers as ps

# df0 = pd.read_pickle("benchmarks/solve_time_2w_10d_with_internal_solver.pkl")
# df1 = pd.read_pickle("benchmarks/solve_time_2w_10d_with_glpk.pkl")

def id_generator(n: int = 3):
    variable_ids = list("abcdefghjklmnopqrstuvwxyz")
    return itertools.permutations(variable_ids, n)

def proposition_generator(depth: int, width: int, variable_iterator):
    if depth <= 0:
        return puan.variable(
            "".join(next(variable_iterator)),
        )
    else:
        return pg.AtLeast(
            1,
            list(
                map(
                    lambda _: proposition_generator(
                        depth=depth-1,
                        width=width,
                        variable_iterator=variable_iterator,
                    ),
                    range(width)
                )
            ),
            puan.variable(
                "".join(next(variable_iterator))
            )
        )

import time
import tqdm
import tm

def test_time_func(func, width: int, depth: int, args = [], kwargs = {}, out = lambda x: x, iter_per_session: int = 100):

    def time_test(width, depth):

        time_obj = {
            'n_depth': depth,
            'n_width': width,
            'size': width**depth*2-1,
        }

        with tm.measure_time("prop_gen", time_obj):
            model = proposition_generator(depth, width, id_generator(depth))

        return {
            **time_obj,
            **out(
                operator.methodcaller(
                    func,
                    *args, 
                    **kwargs
                )(model),
            ),
        }

    time_objs = list(
        map(
            lambda _: time_test(
                width, 
                depth
            ),
            range(iter_per_session)
        )
    )

    return pd.DataFrame(time_objs)

iters = 100
func = "solve_time"
#postfix = ""
postfix = "_with_internal_solver"
# postfix = "_with_glpk"
args = [{}]
kwargs = {
    # "solver": ps.glpk_solver,
}
#out = operator.itemgetter(1)
out = lambda x: x

w=2
for d in [10]:
    t = time.time()
    df = test_time_func(func, w, d, args=args, kwargs=kwargs, out=out, iter_per_session=iters)
    print("")
    print(f"--- Depth {d} ---")
    print(df.mean(axis=0))
    tt = time.time()-t
    print(f"Time per iter: {tt/iters}")
    df.to_pickle(f"benchmarks/{func}_{w}w_{d}d{postfix}.pkl")