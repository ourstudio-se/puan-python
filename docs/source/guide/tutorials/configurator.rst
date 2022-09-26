Create your own Configurator
============================
You've probably seen and tried out a configurator before, for example when buying a product with customization capabilities. One example is car configurators where you select all kinds of different
options, such as color, upholstery, wheels and so on. With each choice you make you get closer to the final product you want to buy. In this
tutorial we will create a configurator using the StingyConfigurator module in Puan and a solver from NpyCVX pypi package.

Wardrobe Wizard using StingyConfigurator
----------------------------------------
In the :ref:`STA-model tutorial<sta-model>` we were modelling a logic system to create an outfit. In this tutorial we will use a simplified version of this example.
The definition of an outfit is the same, but we do not include any hats. The items are reduced to only two t-shirts, two sweaters, two jeans and two pair of shoes.
We define the logic model using the :ref:`PLog modelling system<plog-model>`:

.. code:: python

    import puan
    import puan.logic.plog as pg
    import puan.modules.configurator as cc

    model = cc.StingyConfigurator(
        pg.Xor(
            puan.variable(id="t-thirt-blue"),
            puan.variable(id="t-thirt-black"),
            variable="t-shirts"
        ),
        pg.AtMost(
            propositions=[
                puan.variable(id="sweater-green"),
                puan.variable(id="sweater-black"),
            ],
            value=1,
            variable="sweaters"
        ),
        pg.Xor(
            puan.variable(id="jeans-blue"),
            puan.variable(id="jeans-black"),
            variable="jeans"
        ),
        pg.Xor(
            puan.variable(id="shoes-white"),
            puan.variable(id="shoes-black"),
            variable="shoes"
        ),
        id="outfit"
    )



In other words, we must have exactly one t-shirt, pair of jeans and pair of shoes. The sweater is optional but we can have at most one.
*Note that with these 8 items and 4 rules we have 24 valid outfits*

Now we want to convert our logical model to a linear program in order to efficiently find the best outfit according to some choices, e.g. "I'd like a black outfit".

.. code:: python

    # Convert wardrobe wizard model (logical system) to a polyhedron (linear program) which is used for calculations
    ph = model.to_polyhedron(active=True)

The polyhedron defines the logical model as linear inequalities, such that :math:`A \cdot x \ge b`. For example the ``AtMost`` proposition for sweaters converts to :math:`- sweater\_green - sweater\_black \ge -1`

.. _npycvx: https://github.com/ourstudio-se/puan-npycvx

Before we start to get solutions from certain choices we must create a wrapper for a solver function (a solver is not yet included in Puan).
We use GLPK solver from python package `NpyCVX <npycvx>` and create a wrapper around it, but any solver for linear programs can be used. The `select` function in StingyConfigurator module take
a solver function taking four arguments: the A, and b numpy ndarray's from the polyhedron, which indices are integers and the objective functions:

.. code:: python

    import maz # https://github.com/ourstudio-se/maz-python
    import npycvx # https://github.com/ourstudio-se/puan-npycvx
    import operator
    import functools
    import typing
    import numpy

    def solve_outfit(A, b, int_idxs, objs) -> typing.Iterable[numpy.ndarray]:

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

And now we are ready to select items, and get solutions.

.. code:: python

    # Pick the black pair of jeans
    solution = next(
        model.select(
            {"jeans-black": 1}, 
            solve_outfit,
            only_leafs=True
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
the solutions are equally great. When this is the case, we say that the system is *ambiguous* and can lead to unexpected behaviour. To avoid ambiguity, we use other classes from the StingyConfigurator module
directly.

Notice the "only_leafs" flag. Extra variables are created under the hood and are usually a part of the solution. By setting "only_leafs" to true, we filter out these extra variables.

Fixing ambiguity
----------------
Instead of using the Xor (or Any) class from `puan.logic.plog`, we use them from `puan.modules.configurator` instead since they offer an extra `default` parameter. Now we can define a new configurator model:
(**Notice the cc.Xor instead of pg.Xor**)

.. code:: python

    import puan.logic.plog as pg
    import puan.modules.configurator as cc

    model = cc.StingyConfigurator(
        cc.Xor(
            puan.variable(id="t-thirt-blue"),
            puan.variable(id="t-thirt-black"),
            default="t-thirt-black",
            variable="t-shirts"
        ),
        pg.AtMost(
            propositions=[
                puan.variable(id="sweater-green"),
                puan.variable(id="sweater-black"),
            ],
            value=1,
            variable="sweaters"
        ),
        cc.Xor(
            puan.variable(id="jeans-blue"),
            puan.variable(id="jeans-black"),
            default="jeans-black",
            variable="jeans"
        ),
        cc.Xor(
            puan.variable(id="shoes-white"),
            puan.variable(id="shoes-black"),
            default="shoes-black",
            variable="shoes"
        ),
        id="outfit"
    )

Running the new model, we are guaranteed to get our cool black outfit when none of the other are selected. And if you didn't get the black outfit last run, sure you did get it now.

.. code:: python

    # Pick the black pair of jeans
    solution = next(
        model.select(
            {"jeans-black": 1}, 
            solve_outfit,
            only_leafs=True
        ),
    )
    print(solution)
    # [
    #    (variable(id='jeans-black', dtype=0, virtual=False), 1), 
    #    (variable(id='shoes-black', dtype=0, virtual=False), 1), 
    #    (variable(id='t-thirt-black', dtype=0, virtual=False), 1)
    # ]

More on select
--------------
The `select` function takes a list of "prioritization" dictionaries. They use the key as the id for the selection and a integer value as its prioritization. Lets say you'd like the black jeans and
the black sweater

.. code:: python

    solution = next(
        model.select(
            {
                "jeans-black": 1,
                "sweater-black": 1,
            }, 
            solve_outfit,
            only_leafs=True
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
        pg.AtMost(propositions=["sweater-black", "jeans-black"], value=1)
    )


And solve again solve with same prioritization

.. code:: python

    solution = next(
        new_model.select(
            {
                "jeans-black": 1,
                "sweater-black": 1,
            }, 
            solve_outfit,
            only_leafs=True
        ),
    )
    print(solution)
    # [
    #    (variable(id='jeans-black', dtype=0, virtual=False), 1), 
    #    (variable(id='shoes-black', dtype=0, virtual=False), 1), 
    #    (variable(id='t-thirt-black', dtype=0, virtual=False), 1)
    # ]

And we now did get the black jeans and got rid of our sweater. The reason for this is that the solution with jeans has three items whereas the solution with a sweater has four, a solution less amount of items
is more prioritized than a high number of items. If we increase the prioritization of the sweater, we'll instead get the black sweater with another pair of jeans:

.. code:: python

    # Pick the black pair of jeans
    solution = next(
        new_model.select(
            {
                "jeans-black": 1,
                "sweater-black": 2,
            }, 
            solve_outfit,
            only_leafs=True
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

    solution = next(
        model.select(
            {
                "shoes-black": -1,
                "jeans-black": 1,
                "sweater-black": 2,
            }, 
            solve_outfit,
            only_leafs=True
        ),
    )
    print(solution)
    # [
    #    (variable(id='jeans-blue', dtype=0, virtual=False), 1), 
    #    (variable(id='shoes-white', dtype=0, virtual=False), 1), 
    #    (variable(id='sweater-black', dtype=0, virtual=False), 1), 
    #    (variable(id='t-thirt-black', dtype=0, virtual=False), 1)
    # ]