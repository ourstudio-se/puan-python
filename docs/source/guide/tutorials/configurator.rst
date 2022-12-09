.. _CTUT:

Create your own Configurator
============================
You've probably seen and tried out a configurator before, for example when buying a product with customization capabilities. One example is car configurators where you select all kinds of different
options, such as color, upholstery, wheels and so on. With each choice you make you get closer to the final product you want to buy. In this
tutorial we will create a configurator using the StingyConfigurator module in Puan.

Wardrobe Wizard using StingyConfigurator
----------------------------------------
.. image:: images/release-1.jpg

In this tutorial we will create a model for creating outfits from a set of given items.
The definition of an outfit is to have one and only one t-shirt, pair of jeans and pair of shoes and at most one sweater.
We define the logic model using the :ref:`PLog modelling system<PLOG>`:

.. code:: python

    import puan
    import puan.logic.plog as pg
    import puan.modules.configurator as cc
    import puan_solvers as ps # pip install puan-solvers

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

*Note that with these 8 items and 4 rules we have 24 valid outfits*

Now we want to convert our logical model to a linear program in order to efficiently find the best outfit according to some choices, e.g. "I'd like a black outfit".

.. code:: python

    # Convert wardrobe wizard model (logical system) to a polyhedron (linear program) which is used for calculations
    ph = model.to_ge_polyhedron(active=True)

The polyhedron defines the logical model as linear inequalities, such that :math:`A \cdot x \ge b`. For example the ``AtMost`` proposition for sweaters converts to :math:`- \text{sweater_green} - \text{sweater_black} \ge -1`

.. code:: python

    # Pick the black pair of jeans
    solution = next(
        model.select(
            {"jeans-black": 1}, 
            solver=ps.glpk_solver,
            only_leafs=True # <- only leafs excludes any artificial variable value, which may not be interesting for you as a user
        )
    )

    print(solution)
    # {
    #     'jeans-black': 1, 
    #     'jeans-blue': 0, 
    #     'shoes-black': 1, 
    #     'shoes-white': 0, 
    #     'sweater-black': 0, 
    #     'sweater-green': 0, 
    #     't-thirt-black': 1, 
    #     't-thirt-blue': 0
    # }


We get our black jeans along with black shoes, black t-shirt and no sweater. Seams resonable. But... it could be the case that you didn't get the same solution. Sure, you did get
the black jeans but did you also get the black shoes and t-shirt? It raises an important question: can we guarantee that we will always get the same solution given the same input? 
As it is defined right now, the answer is no. If we'd change to `shoes-white` in our solution, the objective function would return the same objective value, meaning
the solutions are equally good. When this is the case, we say that the system is *ambiguous* and can lead to unexpected behaviour.

Fixing ambiguity
----------------
Instead of using the Xor (or Any) class from :ref:`puan.logic.plog<PLOG>`, we use those from :ref:`puan.modules.configurator<CONF>` instead since they offer an extra `default` parameter. Now we can define a new configurator model:
(**Notice the cc.Xor instead of pg.Xor**)

.. code:: python

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
            solver=ps.glpk_solver,
            only_leafs=True
        ),
    )
    print(solution)
    # {
    #     'jeans-black': 1, 
    #     'jeans-blue': 0, 
    #     'shoes-black': 1, 
    #     'shoes-white': 0, 
    #     'sweater-black': 0, 
    #     'sweater-green': 0, 
    #     't-thirt-black': 1, 
    #     't-thirt-blue': 0
    # }

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
            solver=ps.glpk_solver,
            only_leafs=True
        ),
    )
    print(solution)
    # {
    #     'jeans-black': 1, 
    #     'jeans-blue': 0, 
    #     'shoes-black': 1, 
    #     'shoes-white': 0, 
    #     'sweater-black': 1, 
    #     'sweater-green': 0, 
    #     't-thirt-black': 1, 
    #     't-thirt-blue': 0
    # }

But here both are set to have the same priority. Let's add another logic relationship saying that they cannot be selected together:

.. code:: python

    new_model = model.add(
        pg.AtMost(propositions=["sweater-black", "jeans-black"], value=1)
    )


And solve again with same prioritization

.. code:: python

    solution = next(
        new_model.select(
            {
                "jeans-black": 1,
                "sweater-black": 1,
            }, 
            solver=ps.glpk_solver,
            only_leafs=True
        ),
    )
    print(solution)
    # {
    #     'jeans-black': 1, 
    #     'jeans-blue': 0, 
    #     'shoes-black': 1, 
    #     'shoes-white': 0, 
    #     'sweater-black': 0, 
    #     'sweater-green': 0, 
    #     't-thirt-black': 1, 
    #     't-thirt-blue': 0
    # }

And we now did get the black jeans but not the black sweater. The reason for this is that the solution with jeans has three items whereas the solution with a sweater has four, a solution less amount of items
is more prioritized than a high number of items. If we increase the prioritization of the sweater, we'll instead get the black sweater with another pair of jeans:

.. code:: python

    # Pick the black pair of jeans
    solution = next(
        new_model.select(
            {
                "jeans-black": 1,
                "sweater-black": 2,
            }, 
            solver=ps.glpk_solver,
            only_leafs=True
        ),
    )
    print(solution)
    # {
    #     'jeans-black': 0, 
    #     'jeans-blue': 1, 
    #     'shoes-black': 1, 
    #     'shoes-white': 0, 
    #     'sweater-black': 1, 
    #     'sweater-green': 0, 
    #     't-thirt-black': 1, 
    #     't-thirt-blue': 0
    # }

You can also select with **negative prio**. For instance, you could go with any shoes but the black ones:

.. code:: python

    solution = next(
        model.select(
            {
                "shoes-black": -1,
                "jeans-black": 1,
                "sweater-black": 2,
            }, 
            solver=ps.glpk_solver,
            only_leafs=True
        ),
    )
    print(solution)
    # {
    #     'jeans-black': 0, 
    #     'jeans-blue': 1, 
    #     'shoes-black': 0, 
    #     'shoes-white': 1, 
    #     'sweater-black': 1, 
    #     'sweater-green': 0, 
    #     't-thirt-black': 1, 
    #     't-thirt-blue': 0
    # }