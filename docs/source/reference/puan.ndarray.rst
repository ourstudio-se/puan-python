Numpy Array objects
===================


Summary
--------

Data types
----------

:class:`puan.variable_ndarray`
   A numpy.ndarray sub class which ties variables to the indices of the ndarray.

:class:`puan.ge_polyhedron`
   A numpy.ndarray sub class and a system of linear inequalities forming
   a polyhedron. The "ge" stands for "greater or equal" (:math:`\ge`)
   which represents the relation between :math:`A` and :math:`b` (as in :math:`Ax \ge b`), i.e.
   polyhedron :math:`P=\{x \in R^n \ |\  Ax \ge b\}`.

:class:`puan.ge_polyhedron.config`
   A puan.ge_polyhedron sub class with specific configurator features.

:class:`puan.integer_ndarray`
   A numpy.ndarray sub class with only integers in it.

:class:`puan.boolean_ndarray`
   A numpy.ndarray sub class with only booleans.

.. currentmodule:: puan.ndarray

Variable ndarray
++++++++++++++++
.. autoclass:: variable_ndarray
   :members:
   :undoc-members:
   :show-inheritance:

Ge polyhedron
+++++++++++++
.. autoclass:: ge_polyhedron
   :members:
   :undoc-members:
   :show-inheritance:

Ge polyhedron config
++++++++++++++++++++
.. autoclass:: ge_polyhedron_config
   :members:
   :undoc-members:
   :show-inheritance:

Integer ndarray
+++++++++++++++
.. autoclass:: integer_ndarray
   :members:
   :undoc-members:
   :show-inheritance:

Boolean ndarray
++++++++++++++++
.. autoclass:: boolean_ndarray
   :members:
   :undoc-members:
   :show-inheritance:
