Quickstart
==========

Logic
-----

The fundamental of the logic package is logic models. This is how you create a logic proposition

.. code:: python

    import puan.logic.plog as pl

    # Whenever a string x is passed as a proposition,
    # it will default into a puan.variable(id=x, bounds=(0,1)).
    model = pl.Any("x","y",variable="A")

Propositions can also be created from JSON or tuple. See the :ref:`API reference<proposition>` for details on the formats.
A proposition can also be converted to those formats. 


Linear algebra
--------------

Polyhedron is the most central object in this part of Puan. It's a ``numpy.ndarray`` subclass and represents a collection of linear equations forming a polyhedron. 
This is how you create a polyhedron representing the linear inequality :math:`x + y + z \ge 2`

.. code:: python

    import numpy as np
    import puan.ndarray as nd
    ph = nd.ge_polyhedron(np.array([[2, 1, 1, 1]]), ["#b", "x", "y", "z"])

Configurator
------------

The :class:`puan.modules.configurator.StingyConfigurator` collects and combines the neccessary tools from the logic and linear algebra packages to create a complete configurator.
This is how you create a configurator

.. code:: python

    import puan.modules.configurator as cc
    import puan.logic.plog as pg

    configurator = cc.StingyConfigurator(cc.Any("x","y",variable="A"))

    # View the polyhedron 
    configurator.ge_polyhedron.variables
    configurator.ge_polyhedron
    # Output :
    #    variables              #b   A   x   y
    #    ge_polyhedron_config([[ 1,  1,  0,  0],
    #                          [ 0, -1,  1,  1]]))

    # Get a solution
    solutions = configurator.select({})
    solution_1 = solutions[0]
    solution_1_variable_values = solution_1[0]
    # Output
    # solution_1_variable_values = {'A': 1, 'x': 1, 'y': 0}
    # Each solution contains two more fields which we do not cover in this example

    # Add priority to y and get a new solution
    solutions = configurator.select({"y": 1})
    solution_2_variable_values = solutions[0][0]
    # Output
    # solution_2_variable_values = {'A': 1, 'x': 0, 'y': 1}
    # Now we got y in the solution instead of x 
    # since we added the input: prio 1 to variable y (default is 0)

    # If we prefer y over x we can add it to the model when creating the configurator
    configurator = cc.StingyConfigurator(cc.Any("x","y", variable="A", default="y"))

    # View the polyhedron
    configurator.ge_polyhedron
    # Output
    # variables:             #b   A VAR.. x   y
    # ge_polyhedron_config([[ 1,  1,  0,  0,  0],
    #                       [ 0, -1,  1,  0,  1],
    #                       [ 0,  0, -1,  1,  0]])
    # The polyhedron has one more column than before
    # which is due to a support variable handling the default
    #
    # Get a solution
    solutions = configurator.select({})
    solution_3_variable_values = solutions[0][0]
    # Output solution_3_variable_values =
    #    {'A': 1,
    #     'VARa94110f0d8bb5f16ce1239c8b4163962481545c104501daff7907979dff35024': 0,
    #     'x': 0,
    #     'y': 1}
    # Here we see the support variable in the solution as well, id 'VARa94110...'
    # To omit support variables in the output, set only_leafs to true
    solutions = configurator.select({}, only_leafs=True)
    solution_4_variable_values = solutions[0][0]
    # Output: [{'x': 0, 'y': 1}]
    # This option also filters out variable A since it is dependent on x and y.
    # This makes it a support variable according to our definition
    #
    # The configurator can be extended with more logic using the add function
    configurator.add(pg.Imply(pg.All("y"), cc.Xor("q", "r", default="q")))
    # See all available logic classes and how to use them in the API reference


