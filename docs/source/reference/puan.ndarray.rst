Numpy Array objects
===================


Summary
--------

Data types
----------

.. currentmodule:: puan.ndarray

:class:`variable_ndarray`
   A :class:`numpy.ndarray` sub class which ties variables to the indices of the :class:`numpy.ndarray`.

:class:`ge_polyhedron`
   A :class:`numpy.ndarray` sub class and a system of linear inequalities forming
   a polyhedron. The "ge" stands for "greater or equal" (:math:`\ge`)
   which represents the relation between :math:`A` and :math:`b` (as in :math:`Ax \ge b`), i.e.
   polyhedron :math:`P=\{x \in R^n \ |\  Ax \ge b\}`.

:class:`ge_polyhedron_config`
   A :class:`ge_polyhedron` sub class with configurator features.

:class:`integer_ndarray`
   A :class:`numpy.ndarray` sub class with only integers.

:class:`boolean_ndarray`
   A :class:`numpy.ndarray` sub class with only booleans.


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
