.. _plog-model:

Learn the PLog modelling system
===============================
In this tutorial we're going to learn about the :ref:`Propositional Logic<PLOG>` (PLog for short) modelling system. 
After this tutorial you will know how to create propositional logical expressions and how to apply different tools to them. We will briefly go through
the fundamental classes and given some intuitive examples.


The Fridge Example
------------------
A logical system can be constructed or applied in a vast variety of fields. In this example we create a logical system
for what to buy in the grocery store given the containments of our fridge. Those are the instructions

- if the milk is less than half full, then buy new pack of milk
- always buy a small bag of chips (we eat a lot of chips)
- if tomatos are less than or equal to 3, then buy more
- if no cucumber, then buy more
- the amount of tomatoes and cucumbers should not be more than 5 each

To model this, we need first to recognize our variables and their type (boolean or integer).

- **milk_half**     : bool   - since saying if it is less than half full (in fridge) is a yes or no question
- **milk_bought**   : bool   - since saying if it is milk was bought is a yes or no question
- **chips**         : bool   - just says bag of chips, no quantity
- **tomatoes**      : int    - they are restricted in a range between 4-5
- **cucumbers**     : int    - they are restricted in a range between 1-5

Let's see what those definitions looks like in code

.. code:: python

    import puan

    milk_home   = puan.variable(id="milk_home")
    milk_bought = puan.variable(id="milk_bought")
    chips       = puan.variable(id="chips")
    tomatoes    = puan.variable(id="tomatoes",   dtype="int")
    cucumbers   = puan.variable(id="cucumbers",  dtype="int")

Now we need to define the logical relationships between the items. We start by taking a look at the **milk**. 
We were waying that if the milk in the fridge is half full, i.e milk_home is False or not milk_home is True, then it is implied that also the milk is bought, i.e milk_bought is True. 
This is called an implication:

.. math::

   \text{milk_done_right} = \text{not milk_home} \rightarrow \text{milk_bought} 
   
There is a proposition class for this expression called :class:`Imply<puan.logic.plog.Imply>` which models an implication. 
The implication-proposition has two properties; a condition and a consequence where imply is the relation between the condition and the consequence. 
For instance, we can define *if milk is half* (condition), then buy more milk (consequence). 

Let's put it into code.

.. code:: python

    import puan.logic.plog as pg

    milk_done_right = pg.Imply(
        condition=pg.Not(milk_home),
        consequence=milk_bought,
        variable="milk_done_right"
    )

The number of tomatoes and cucumbers must not be larger than 5 each, and the number of tomatoes should not be less than 2.
We model this using the proposition classes :class:`AtLeast<puan.logic.plog.AtLeast>` and :class:`AtMost<puan.logic.plog.AtMost>`.
Those classes takes :class:`proposition<puan.Proposition>` as input together with a ``value`` defining the upper and lower bound of the propostion respectively.  

.. code:: python

    # tomatoes greater or equal to 4
    tomatoes_ge_four = pg.AtLeast(propositions=[tomatoes], value=4, variable="tomatoes_ge_four")

    # tomatoes and cucumbers less or equal to 5
    tomatoes_le_five = pg.AtMost(propositions=[tomatoes], value=5, variable="tomatoes_le_five")
    cucumbers_le_five = pg.AtMost(propositions=[cucumbers], value=5, variable="cucumbers_le_five")

    # cucumbers greater or equal to 1 
    cucumbers_ge_one = pg.AtLeast(propositions=[cucumbers], value=1, variable="cucumbers_ge_one")
    
Now, if all of these variables are true, then it means that number of tomatoes is between 4-5 and number of cucumbers is between 1-5.
To tie these two expressions we need to plug them into a so called All-proposition.
*Note that the All-proposition is a special case of the AtLeast-proposition*.

.. code:: python

    vegetables_ok = pg.All(
        tomatoes_ge_four,
        tomatoes_le_five,
        cucumbers_le_five,
        cucumbers_ge_one,
        variable="vegetables"
    )

Now we can put it all together in a single plog-model

.. code:: python

    fridge_model = pg.All(
        chips,
        milk_done_right,
        vegetables_ok,
        variable="fridge"
    )

*Note how we can create propositions by combining booleans like chips_is_true with more advanced propositions, such as the vegetables_ok, to create a logical system*.

Now it's time to see what we have in the fridge:

- milk is less than half full
- we have two tomatoes and no cucumbers

We head to the store and check our model with the current shopping cart after we added two tomatoes:

.. code:: python

    cart = {
        milk_home.id: 1,
        milk_bought.id: 0,
        tomatoes.id: 2+2,
        cucumbers.id: 0,
    }

    # ... and evaluate if it satisfies the model
    print(fridge_model.evaluate(cart))
    # >>> Bounds(lower=0, upper=0) 
    # Meaning that this cart evaluates fridge_model to a constant 0 (or false)

As expected, the current cart is not valid (we don't have *chips* nor *cucumbers*). Let's pick them from the store and
check again if we're ok

.. code:: python

    # Construct a cart numpy array instance from variables ...
    new_cart = {
        chips.id: 1,
        milk_home.id: 1,
        milk_bought.id: 0,
        tomatoes.id: 2+2,
        cucumbers.id: 1,
    }

    # ... and evaluate if it satisfies the model
    print(fridge_model.evaluate(new_cart))
    # >>> Bounds(lower=1, upper=1)
    # Meaning that this cart evaluates fridge_model to a constant 1 (or true)

The model is satisfied and we are ready to checkout and go home.