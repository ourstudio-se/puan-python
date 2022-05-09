Condition-Implies-Consequence
=============================

Summary
--------

Condition-Implies-Consequence (cic) is an abstract data type defining logical relationships
between variables in a combinatorial optimization manner. A cic is created from "if this then that"
sentence and is easy to grasp. It is also a specific instance from a propositional
logic expression with the implies-operator in between a "if" and "then".
For example, "if it is raining then I'll take the umbrella" could be written as :math:`a \rightarrow b` where
:math:`a =` "it is raining" and :math:`b =` "take the umbrella".

Data types
----------
   :class:`puan.logic.cic.proposition` : A proposition is an abstract class and a logical object that can be resolved into a true or false value.
   
   :class:`puan.logic.cic.variable_proposition` : A variable_proposition is a logical object that can be resolved into a true or false value.
   
   :class:`puan.logic.cic.boolean_variable_proposition` : A boolean_variable_proposition is a logical object that can be resolved into a true or false value. The boolean_variable_proposition has a variable and a value it is expected to have.

   :class:`puan.logic.cic.discrete_variable_proposition` : A discrete_variable_proposition is a logical object that can be resolved into a true or false value. The discrete_variable_proposition has a variable, an operator and a value. The variable dtype will be forced into an int and the value must be an int. The expression (x >= 1) is considered a discrete variable proposition.
   
   :class:`puan.logic.cic.conditional_proposition` : A conditional_proposition is a logical object that can be resolved into a true or false value. The conditional_proposition has a relation and a list of propositions. There are two relation types (ALL/ANY) and the proposition will be considered true if either ALL or ANY of its propositions are true (depending if relation is ALL or ANY).
   
   :class:`puan.logic.cic.consequence_proposition` : A consequence_proposition is a logical object that can be resolved into a true or false value. The consequence_proposition is a conditional_proposition with the exception of an extra field "default". This is used to mark which underlying propositions is default if many are considered equally correct.
   
   :class:`puan.logic.cic.Implication` : Enum class of what implication methods exists. 
   
   :class:`puan.logic.cic.implication_proposition` : A implication_proposition is a logical object that can be resolved into a true or false value. The implication_proposition has a logical structure of condition - implies -> consequence, or the more common sentence "if this then that". In other words, the proposition is false only if the condition is considered true while the consequence is false.
   
   :class:`puan.logic.cic.conjunctional_proposition` : A conjunctional_proposition is a logical object that can be resolved into a true or false value. The conjunctional_proposition is a conditional_proposition with the relation type set to ALL.

   :class:`puan.logic.cic.cicR` : The RAW format of a cic, meaning the condition and consequence are both conjunctions. This format has a one-to-one mapping into a linear programming constraint.

   :class:`puan.logic.cic.cicJE` : JSON version


.. currentmodule:: puan.logic.cic

boolean_variable_proposition
++++++++++++++++++++++++++++
.. autoclass:: boolean_variable_proposition
   :members:
   :show-inheritance:

discrete_variable_proposition
+++++++++++++++++++++++++++++
.. autoclass:: discrete_variable_proposition
   :members:
   :show-inheritance:

conditional_proposition
+++++++++++++++++++++++
.. autoclass:: conditional_proposition
   :members:
   :show-inheritance:

consequence_proposition
+++++++++++++++++++++++
.. autoclass:: consequence_proposition
   :members:
   :show-inheritance:

Implication
+++++++++++
.. autoclass:: Implication
   :members:
   :show-inheritance:

implication_proposition
+++++++++++++++++++++++
.. autoclass:: implication_proposition
   :members:
   :show-inheritance:

conjunctional_proposition
+++++++++++++++++++++++++
.. autoclass:: conjunctional_proposition
   :members:
   :show-inheritance:

cicR
+++++

.. autoclass:: cicR
   :members:
   :show-inheritance:

.. _cicJE:

cicJE
+++++

.. autoclass:: cicJE
   :members:
   :show-inheritance: