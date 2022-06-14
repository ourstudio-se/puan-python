Propositional Logic
===================

Summary
--------

Propositional Logic (plog) system is an abstract data type defining logical relationships between variables in a combinatorial optimization manner.

Data types
----------
   :class:`puan.logic.plog.Proposition` : A Proposition is an abstract class and an extension of the puan.variable class with the intention of being the atomic version of a compound proposition. When evaluated, the Proposition will take on a value and is later at some point evaluated to either true or false.

   :class:`puan.logic.plog.AtLeast` : AtLeast is a compound proposition which takes propositions and represents a lower bound on the result of those propositions. For example, select at least one of x, y and z would be defined as AtLeast("x","y","z", value=1) and represented as x+y+z >= 1.

   :class:`puan.logic.plog.AtMost` : AtMost is a compound proposition which takes propositions and represents a lower bound on the result of those propositions. For example, select at least one of x, y and z would be defined as AtMost("x","y","z", value=2) and represented as -x-y-z >= -2.
   
   :class:`puan.logic.plog.All` : 'All' is a compound proposition representing a conjunction of all given propositions. 'All' is represented by an AtLeast -proposition with value set to the number of given propositions. For example, x = All("x","y","z") is the same as y = AtLeast("x","y","z",value=3) 
   
   :class:`puan.logic.plog.Any` : 'Any' is a compound proposition representing a disjunction of all given propositions. 'Any' is represented by an AtLeast -proposition with value set to 1. For example, Any("x","y","z") is the same as AtLeast("x","y","z",value=1) 

   :class:`puan.logic.plog.Imply` : Imply is the implication logic operand and has two main inputs: condition and consequence. For example, if x is selected then y and z must be selected could be defined with the Imply -class as Imply("x", All("y","z")). 
   
   :class:`puan.logic.plog.Xor` : Xor is restricting all propositions within to be selected exactly once. For example, Xor("x","y","z") means that x, y and z must be selected exactly once.
   
   :class:`puan.logic.plog.Not` : Not is restricting propositions to never be selected. For example, Not("x","y","z") means that x, y or z can never be selected. Note that Not(x) is not necessarily equivilent to x.invert() (but could be).


.. currentmodule:: puan.logic.plog

Proposition
+++++++++++
.. autoclass:: Proposition
   :members:
   :inherited-members:
   :show-inheritance:

AtLeast
+++++++
.. autoclass:: AtLeast
   :members:
   :inherited-members:
   :show-inheritance:

AtMost
++++++
.. autoclass:: AtMost
   :members:
   :inherited-members:
   :show-inheritance:

All
+++
.. autoclass:: All
   :members:
   :show-inheritance:

Any
+++
.. autoclass:: Any
   :members:
   :show-inheritance:

Imply
+++++
.. autoclass:: Imply
   :members:
   :show-inheritance:

Xor
+++
.. autoclass:: Xor
   :members:
   :show-inheritance:

Not
+++
.. autoclass:: Not
   :members:
   :show-inheritance:

