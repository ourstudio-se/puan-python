.. _PLOG:

Propositional Logic
===================

Summary
--------

Propositional Logic (plog) system is an abstract data type defining logical relationships between variables in a combinatorial optimization manner.

Data types
----------

   :class:`puan.logic.plog.AtLeast` : ``AtLeast`` is a compound proposition which takes propositions and represents a lower bound on the result of those propositions. For example, select at least one of x, y and z would be defined as ``AtLeast(propositions=["x","y","z"], value=1)`` and represented by the linear inequality :math:`x+y+z \ge 1`.

   :class:`puan.logic.plog.AtMost` : ``AtMost`` is a compound proposition which takes propositions and represents a lower bound on the result of those propositions. For example, select at most two of x, y and z would be defined as ``AtMost(propositions=["x","y","z"], value=2)`` and represented by the linear inequality :math:`-x-y-z \ge -2`.
   
   :class:`puan.logic.plog.All` : ``All`` is a compound proposition representing a conjunction of all given propositions. ``All`` is represented by an ``AtLeast`` proposition with value set to the number of given propositions. For example, ``All("x","y","z")`` is equivalent to ``AtLeast(propositions=["x","y","z"], value=3)``.
   
   :class:`puan.logic.plog.Any` : ``Any`` is a compound proposition representing a disjunction of all given propositions. ``Any`` is represented by an ``AtLeast`` proposition with value set to 1. For example, ``Any("x","y","z")`` is equivalent to ``AtLeast(propositions=["x","y","z"], value=1)``.

   :class:`puan.logic.plog.Imply` : ``Imply`` is the implication logic operand and has two main inputs: condition and consequence. For example, if x is selected then y and z must be selected could be defined with the ``Imply`` class as ``Imply("x", All("y","z"))``. 
   
   :class:`puan.logic.plog.Xor` : ``Xor`` is restricting all propositions within to be selected exactly once. For example, ``Xor("x","y","z")`` means that one and exactly one of x, y and z must be selected.
   
   :class:`puan.logic.plog.XNor` : ``XNor`` is a negated ``Xor``. In the special case of two propositions, this is equivalent to the biconditional logical connective (<->).
   
   :class:`puan.logic.plog.Not` : ``Not`` is negating a proposition. For example, ``Not(All("x","y","z"))`` means that x, y and z can never be selected all together and is equivalent to ``AtMost(propositions=["x","y","z"], value=2)``.


.. currentmodule:: puan.logic.plog

AtLeast
+++++++
.. autoclass:: AtLeast
   :members:
   :undoc-members:
   :show-inheritance:

AtMost
++++++
.. autoclass:: AtMost
   :members:
   :undoc-members:
   :show-inheritance:

All
+++
.. autoclass:: All
   :members:
   :undoc-members:
   :show-inheritance:

Any
+++
.. autoclass:: Any
   :members:
   :undoc-members:
   :show-inheritance:

Imply
+++++
.. autoclass:: Imply
   :members:
   :undoc-members:
   :show-inheritance:

Xor
+++
.. autoclass:: Xor
   :members:
   :undoc-members:
   :show-inheritance:

XNor
++++
.. autoclass:: XNor
   :members:
   :undoc-members:
   :show-inheritance:

Not
+++
.. autoclass:: Not
   :members:
   :undoc-members:
   :show-inheritance:

PropositionValidationError
++++++++++++++++++++++++++
.. autoclass:: PropositionValidationError
   :members:
   :undoc-members:
   :show-inheritance: