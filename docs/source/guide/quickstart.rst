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

TODO:Complete example

.. code:: python

    import puan.modules.configurator as cc
    import puan.logic.plog as pl

    configurator = cc.StingyConfigurator(pl.Any("x","y",variable="A"))

    # View the polyhedron 
    configurator.to_ge_polyhedron(True)
    # Output :  ge_polyhedron([[ 1,  1,  0,  0],
    #                          [ 0, -1,  1,  1]]))

    # Get a solution
    configurator.select({})
    # Output 
