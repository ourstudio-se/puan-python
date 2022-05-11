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
   :class:`puan.logic.cic.Operator` : Enum class of what operators exist. 

   :class:`puan.logic.cic.proposition` : A proposition is an abstract class and a logical object that can be resolved into a true or false value.
   
   :class:`puan.logic.cic.variable_proposition` : A variable_proposition is a logical object that can be resolved into a true or false value.
   
   :class:`puan.logic.cic.boolean_variable_proposition` : A boolean_variable_proposition is a logical object that can be resolved into a true or false value. The boolean_variable_proposition has a variable and a value it is expected to have.

   :class:`puan.logic.cic.discrete_variable_proposition` : A discrete_variable_proposition is a logical object that can be resolved into a true or false value. The discrete_variable_proposition has a variable, an operator and a value. The variable dtype will be forced into an int and the value must be an int. The expression (x >= 1) is considered a discrete variable proposition.
   
   :class:`puan.logic.cic.conditional_proposition` : A conditional_proposition is a logical object that can be resolved into a true or false value. The conditional_proposition has a relation and a list of propositions. There are two relation types (ALL/ANY) and the proposition will be considered true if either ALL or ANY of its propositions are true (depending if relation is ALL or ANY).
   
   :class:`puan.logic.cic.conditional_variable_proposition` : A conditional variable proposition is a special type of conditional proposition with the exception of only taking variable propositions as its propositions. 
   
   :class:`puan.logic.cic.conjunctional_variable_proposition` : A conjunctional variable proposition is a special type of conditional variable proposition with the exception of having **ALL** -relation set.
   
   :class:`puan.logic.cic.disjunctional_variable_proposition` : A disjunctional variable proposition is a special type of conditional variable proposition with the exception of having **ANY** -relation set.
   
   :class:`puan.logic.cic.conjunction_normal_form` : A conjunction normal form proposition is a logical object that can be resolved into a true or false value. The class is a special variant of a conditional proposition with but on conjunction normal form, wheras the conditional proposition could take on many forms. A conjunctional proposition has **ALL** -relation and requires all its propositions to be of type disjunctional variable proposition.
   
   :class:`puan.logic.cic.disjunction_normal_form` : A disjunctional normal form proposition is a logical object that can be resolved into a true or false value. The class is a special variant of a conditional proposition nut on disjunction normal form, wheras the conditional proposition could take on many forms. A disjunctional proposition has **ANY** -relation relation and requires all its propositions to be of type conjunctional variable proposition.
   
   :class:`puan.logic.cic.implication_proposition` : A implication_conditional_proposition is a logical object that can be resolved into a true or false value. The implication_conditional_proposition has a logical structure of condition - implies -> consequence, or the more common sentence "if this then that".
   
   :class:`puan.logic.cic.implication_conjunctional_variable_proposition` : A implication_disjunction_normal_form differs from implication_proposition in that it only takes conjunctional_variable_proposition(s) as condition and consequences.
   
   :class:`puan.logic.cic.conjunctional_implication_proposition` : A conjunctional implication proposition is a logical object that can be resolved into a true or false value. The conjunctional implication proposition is a conditional_proposition with the relation type set to ALL and propositions only of type implication propositions.

   :class:`puan.logic.cic.cicR` : The RAW format of a cic, meaning the condition and consequence are both conjunctions. This format has a one-to-one mapping into a linear programming constraint.

   :class:`puan.logic.cic.cicJE` : JSON version


.. currentmodule:: puan.logic.cic

Operator
++++++++
.. autoclass:: Operator
   :members:
   :show-inheritance:


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
   
conditional_variable_proposition
++++++++++++++++++++++++++++++++
.. autoclass:: conditional_variable_proposition
   :members:
   :show-inheritance:
   
conjunctional_variable_proposition
++++++++++++++++++++++++++++++++++
.. autoclass:: conjunctional_variable_proposition
   :members:
   :show-inheritance:
   
disjunctional_variable_proposition
++++++++++++++++++++++++++++++++++
.. autoclass:: disjunctional_variable_proposition
   :members:
   :show-inheritance:

conjunction_normal_form
+++++++++++++++++++++++
.. autoclass:: conjunction_normal_form
   :members:
   :show-inheritance:

disjunction_normal_form
+++++++++++++++++++++++
.. autoclass:: disjunction_normal_form
   :members:
   :show-inheritance:

implication_proposition
+++++++++++++++++++++++
.. autoclass:: implication_proposition
   :members:
   :show-inheritance:

implication_conjunctional_variable_proposition
++++++++++++++++++++++++++++++++++++++++++++++
.. autoclass:: implication_conjunctional_variable_proposition
   :members:
   :show-inheritance:

conjunctional_implication_proposition
+++++++++++++++++++++++++++++++++++++
.. autoclass:: conjunctional_implication_proposition
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