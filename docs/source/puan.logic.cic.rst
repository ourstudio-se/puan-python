puan.logic.cic
===============

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
   :class:`puan.logic.cic.cicR` : The RAW format of a cic, meaning the condition and consequence are both conjunctions. This format has a one-to-one mapping into a linear programming constraint.

   :class:`puan.logic.cic.cicE` : A more Expressive format where the condition can be written either as a DNF or a CNF, also ``REQUIRES_EXCLUSIVELY`` rule type exist here.

   :class:`puan.logic.cic.cicJE` : JSON version of cicE


.. currentmodule:: puan.logic.cic

cicR
+++++

.. autoclass:: cicR
   :members:
   :undoc-members:
   :show-inheritance:

cicE
+++++

.. autoclass:: cicE
   :members:
   :undoc-members:
   :show-inheritance:

cicJE
+++++

.. autoclass:: cicJE
   :members:
   :undoc-members:
   :show-inheritance: