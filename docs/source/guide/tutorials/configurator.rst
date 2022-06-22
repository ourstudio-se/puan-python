Create your own Configurator
============================
You've probably seen and tried out one before. Maybe the most common ones are car configurators where you select all kinds of different
car options, such as color, upholstery, wheels and so on, and seeing your car getting more and more towards how you'd want it. In this
tutorial we will create a simple configurator using the Configurator module from Puan and a solver from NpyCVX pypi package.

Wardrobe Wizard using Configurator
----------------------------------
From a previous tutorial we were modelling a logic system to represent an outfit. Here we will show how to use the system once it has been defined.
For simplicity, we will use only two t-shirts, two sweaters, two jeans and two pair of shoes. We start off by defining or logic model based on these requirements:

.. code:: python

    import puan.logic.plog as pg
    import puan.modules.configurator as cc

    model = cc.Configurator(
        pg.Xor(
            pg.Proposition("t-thirt-blue"),
            pg.Proposition("t-thirt-black"),
            id="t-shirts"
        ),
        pg.AtMost(
            pg.Proposition("sweater-green"),
            pg.Proposition("sweater-black"),
            value=1,
            id="sweaters"
        ),
        pg.Xor(
            pg.Proposition("jeans-blue"),
            pg.Proposition("jeans-black"),
            id="jeans"
        ),
        pg.Xor(
            pg.Proposition("shoes-white"),
            pg.Proposition("shoes-black"),
            id="shoes"
        ),
        id="outfit"
    )


.. _npycvx: https://github.com/ourstudio-se/puan-npycvx

In other words, we must have exactly one t-shirt, pair of jeans and pair of shoes. The sweater is optional but at most one.
Before we start to get solutions from certain choices we must create a wrapper for a solver function (a solver is not yet included in Puan).
We use GLPK solver from python package `NpyCVX <npycvx>` and create a wrapper around it. The `select` function in Configurator module take
a solver function taking four arguments: the A, and b numpy ndarray's from the polyhedron, which indices are integers and the objective functions:

.. code:: python

    import maz # https://github.com/ourstudio-se/maz-python
    import npycvx # https://github.com/ourstudio-se/puan-npycvx
    import operator
    import functools

    def solve_stuff(A, b, int_idxs, objs) -> typing.Iterable[numpy.ndarray]:

        """
            First, prepare, load and convert the polytope into a cvx-object and then start
            to solve all objectives. npycvx.solve_lp returns a tuple of (status, solution) which we
            from it gets the second item.
        """

        return map(
            maz.compose( # <- maz is a functional programming package (https://github.com/ourstudio-se/maz-python)
                operator.itemgetter(1), # <- get the second item from solution tuple
                functools.partial(
                    npycvx.solve_lp, # <- solve linear programming problem
                    *npycvx.convert_numpy(A, b, set(int_idxs)), # <- prepare problems and set which indices are ints
                    False # <- minimize = False, i.e. we set problem to be maximized
                )
            ),
            objs
        )

And now we are ready to select stuff, and get solutions.

.. code:: python

    # Pick the black pair of jeans
    solution = list(
        model.select(
            [{"jeans-black": 1}], 
            solve_stuff
        )
    )

    print(solution)
    # [
    #    (variable(id='jeans-black', dtype=0, virtual=False), 1), 
    #    (variable(id='shoes-black', dtype=0, virtual=False), 1), 
    #    (variable(id='t-thirt-black', dtype=0, virtual=False), 1)
    # ]


We get our black jeans along with black shoes, black t-shirt and no sweater. Seams resonable. But... it could be the case that you didn't get the same solution. Sure, you did get
the black jeans but did you also get the black shoes and t-shirt? It raises an important question: can we guarantee that we will always get the same solution given the same input? 
As it is defined right now, the answer is no. Well, to be exact, the answer is actually yes but that's not the point. If we'd change to `shoes-white` in our solution, the objective function would return the same objective value, meaning
the solutions are equally great. When this is the case, we say that the system is *ambiguous* and can lead to unexpected behaviour in the end. To avoid ambiguity, we use other classes from the Configurator module
directly.

Fixing ambiguity
----------------
Instead of using the Xor (or Any) class from `puan.logic.plog`, we use them from `puan.modules.configurator` instead since they offer an extra `default` parameter. Now we can define a new configurator model:
(**Notice the cc.Xor instead of pg.Xor**)

.. code:: python

    import puan.logic.plog as pg
    import puan.modules.configurator as cc

    model = cc.Configurator(
        cc.Xor(
            pg.Proposition("t-thirt-blue"),
            pg.Proposition("t-thirt-black"),
            default="t-thirt-black",
            id="t-shirts"
        ),
        pg.AtMost(
            pg.Proposition("sweater-green"),
            pg.Proposition("sweater-black"),
            value=1,
            id="sweaters"
        ),
        cc.Xor(
            pg.Proposition("jeans-blue"),
            pg.Proposition("jeans-black"),
            default="jeans-black",
            id="jeans"
        ),
        cc.Xor(
            pg.Proposition("shoes-white"),
            pg.Proposition("shoes-black"),
            default="shoes-black",
            id="shoes"
        ),
        id="outfit"
    )

Running the new model, we are guaranteed to get our cool black outfit when none of the other are selected. And if you didn't get the black outfit last run, sure you did get it now.

.. code:: python

    # Pick the black pair of jeans
    solution = list(
        model.select([{"jeans-black": 1}], solve_stuff),
    )
    print(solution)
    # [
    #    (variable(id='jeans-black', dtype=0, virtual=False), 1), 
    #    (variable(id='shoes-black', dtype=0, virtual=False), 1), 
    #    (variable(id='t-thirt-black', dtype=0, virtual=False), 1)
    # ]

More on select
--------------
The `select` function takes a list of "prioritization" dictionaries. They use the key as the id for the selection and a integer value as its prioritization. Let say you'd like the black jeans and
the black sweater

.. code:: python

    # Pick the black pair of jeans
    solution = list(
        model.select(
            [
                {
                    "jeans-black": 1,
                    "sweater-black": 1,
                }
            ], 
            solve_stuff
        ),
    )
    print(solution)
    # [
    #    (variable(id='jeans-black', dtype=0, virtual=False), 1), 
    #    (variable(id='shoes-black', dtype=0, virtual=False), 1), 
    #    (variable(id='sweater-black', dtype=0, virtual=False), 1), 
    #    (variable(id='t-thirt-black', dtype=0, virtual=False), 1)
    # ]

But here both are set to have the same priority. Let's add another logic relationship saying that they cannot be selected together:

.. code:: python

    new_model = model.add(
        pg.AtMost("sweater-black", "jeans-black", value=1)
    )


And again solve with same prio

.. code:: python

    # Pick the black pair of jeans
    solution = list(
        model.select(
            [
                {
                    "jeans-black": 1,
                    "sweater-black": 1,
                }
            ], 
            solve_stuff
        ),
    )
    print(solution)
    # [
    #    (variable(id='jeans-black', dtype=0, virtual=False), 1), 
    #    (variable(id='shoes-black', dtype=0, virtual=False), 1), 
    #    (variable(id='t-thirt-black', dtype=0, virtual=False), 1)
    # ]

And we know did get the black jeans and got rid of our sweater. The reason for this is that the solution with jeans has 3 items whereas the solution with a sweater has 4 and a low number of items
are more prioritized than a high number of items. If we change sweater prio to be higher than the jeans, we'll instead get the black sweater with another pair of jeans:

.. code:: python

    # Pick the black pair of jeans
    solution = list(
        model.select(
            [
                {
                    "jeans-black": 1,
                    "sweater-black": 2,
                }
            ], 
            solve_stuff
        ),
    )
    print(solution)
    # [
    #    (variable(id='jeans-blue', dtype=0, virtual=False), 1), 
    #    (variable(id='shoes-black', dtype=0, virtual=False), 1), 
    #    (variable(id='sweater-black', dtype=0, virtual=False), 1), 
    #    (variable(id='t-thirt-black', dtype=0, virtual=False), 1)
    # ]

You can also select with **negative prio**. For instance, you could go with any shoes but the black ones:

.. code:: python

    # Pick the black pair of jeans
    solution = list(
        model.select(
            [
                {
                    "shoes-black": -1,
                    "jeans-black": 1,
                    "sweater-black": 2,
                }
            ], 
            solve_stuff
        ),
    )
    print(solution)
    # [
    #    (variable(id='jeans-blue', dtype=0, virtual=False), 1), 
    #    (variable(id='shoes-white', dtype=0, virtual=False), 1), 
    #    (variable(id='sweater-black', dtype=0, virtual=False), 1), 
    #    (variable(id='t-thirt-black', dtype=0, virtual=False), 1)
    # ]