What is Puan?
=============

Puan combines the fields of logic and linear programming, compiling to a configurator.

Logic
-----

Puan makes it easy to define and manage logical models. Two modelling approaches are available, the generic :ref:`source-target application model (STA)<sta-model>` and the classical :ref:`propositional logic model (PLog)<plog-model>`.
The STA model allows to define high level relations between items, without specifying the items explicitly. The PLog model allows to define any propositional logic and its combinations.
The Puan package also comes with tools to manage logical systems. *Reduction* is one tool which examines the ruleset and if the ruleset forces a certain value to a variable we let the varaible get that value and remove the variable from the ruleset.
Another tool is *assume* where you assign a value to one or several items, this action is follwed by a *reduction*.

Conversion to polyhedron
------------------------

An STA or PLog model is easy to define, but hard to make computations on. However, a proposition could be represented by a linear constraint, e.g. ``Any(a, b, c)`` could be represented by the linear inequality :math:`a + b + c \ge 1`.
Converting the entire logical system gives a collection of linear inequalities, i.e. a polyhedron. Being in this domain makes it easier to perform calculations as the tools from linear algebra becomes available. 

Linear programming
------------------

Puan does not have a solver of its own at this point, but contains various of functions and tools to setup and preprocess a linear program. 


Configurator
------------

A configurator could be constructed by the collection of Puan tools. The logical model describes the supply and allowed combinations of what is to be configured. Converting the logical model to a polyhedron
makes it possible to find configurations by solving a linear program, which in general is more efficient than solving a logic program. The last thing to add to the configurator is the objective function, i.e. the behaviour of the configurator.
Puan provides different methods resulting in different behaviours of the configurator, e.g. last option is the most prioritized or all options are equally prioritized. 
