Configurator module
===================

Summary
-------
The Configurator module helps to easily create an online configurator experience from a logic any logic model. 

Data types
----------

:class:`puan.modules.configurator.Any` : Overriding :class:`puan.logic.plog.Any` class to take extra parameter ``default``.

:class:`puan.modules.configurator.Xor` : Overriding :class:`puan.logic.plog.Xor` class to take extra parameter ``default``.

:class:`puan.modules.configurator.StingyConfigurator` : A class for supporting a sequential configurator experience. The "stingy" configurator always tries to select the least possible number of selections in a solution, with respect to what's been prioritized. Whenever an ``AtLeast`` proposition proposes multiple solutions that equal least number of selections, then a default may be added to avoid ambivalence.

.. currentmodule:: puan.modules.configurator

Any
+++
.. autoclass:: Any
   :members:
   :undoc-members:
   :show-inheritance:

Xor
+++
.. autoclass:: Xor
   :members:
   :undoc-members:
   :show-inheritance:

StingyConfigurator
++++++++++++++++++
.. autoclass:: StingyConfigurator
   :members:
   :undoc-members:
   :show-inheritance: