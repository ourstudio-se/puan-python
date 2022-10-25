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

Propositions can also be created from JSON, tuple or dict. See the :ref:`API reference<API>` for details on the formats.
A proposition can also be converted to those formats. 


Linear algebra
--------------

Polyhedron is the most central object in this part of Puan. It's a numpy.ndarray subclass and represents a collection of linear equations forming a polyhedron. 
This is how you create a polyhedron representing the linear inequality :math:`x + y + z \ge 2`

.. code:: python

    import numpy as np
    import puan.ndarray as nd
    ph = nd.ge_polyhedron(np.array([[2, 1, 1, 1]]), ["#b", "x", "y", "z"])