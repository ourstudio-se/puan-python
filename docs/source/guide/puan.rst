What is Puan?
=============

Puan is a toolkit for combining the fields of logic and linear programming, with the main purpose to compiling to a configurator.

Logic
-----

Puan makes it easy to define logical models. Models are defined by propositions forming a :ref:`propositional logic model (PLog)<plog-model>`.

Conversion to polyhedron
------------------------

A PLog model is easy to define, but hard to make computations on. However, a proposition could be represented by a linear constraint, e.g. ``Any(a, b, c)`` could be represented by the linear inequality :math:`a + b + c \ge 1`.
Converting the entire logical system gives a collection of linear inequalities, i.e. a polyhedron. Being in this domain makes it easier to perform calculations as the tools from linear algebra becomes available.
*Reduce* is one tool which examines the constraints and if the constraints forces a certain value to a variable we let the varaible get that value and remove it from the polyhedron. Reducing the size of the polyhedron decreases
the complexity of the problem, resulting in faster runtimes.

Linear programming
------------------

Linear programming is the task of finding the opitmal solution to an objective function while satisfying a collection of given constraints.
Puan has a Beta-solver for solving integer linear programs. Note, this solver should not be used in production. The `puan-solvers package <https://github.com/ourstudio-se/puan-solvers>`_ provides easy integration with the most common open-source solvers.


Configurator
------------

A configurator could be constructed by the collection of Puan tools. The logical model describes the items and allowed combinations of what is to be configured. Converting the logical model to a polyhedron
makes it possible to find configurations by solving a linear program, which in general is more efficient than solving a logic program. The last thing to add to the configurator is the objective function, i.e. the behaviour of the configurator.
Puan provides different methods resulting in different behaviours of the configurator, see examples in the :ref:`configuration tutorial<CTUT>`.
