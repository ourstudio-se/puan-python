Source-target Application 
=========================

.. _puan.logic.sta:

Summary
--------
The "source-target-application" data format (`sta` for short) defines logical relations between items bound 
on their properties. Let say you're modelling logical relationships between the entities Person, House and Car, you
could say 
   - a Person has at most one House 
   - a Car and a House belongs to exactly one Person 
without explicitly saying which are the Person's, House's and Car's. The Application instance can later be exported
into a list of :ref:`cicJE <cicJE>`.

Data types
----------

.. currentmodule:: puan.logic.sta

Application
+++++++++++++

.. autoclass:: application
   :members:
   :undoc-members:
   :show-inheritance: