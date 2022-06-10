Learn the CIC logic system
==========================
In this tutorial we're going to learn about the Condition-Implies-Consequence (CIC for short) logic system 
to be able to create propositional logical expressions and use different tools on. We will briefly go through
the fundamental classes while given some intuitive examples.


The Fridge Example
------------------
Let say I knew some logical system and wanted to model it quickly and test if an interpretation of that model
is satisfied or not. For example, my partner and I had created some instructions regarding what to buy given the containments
of our fridge. This was the instructions

- if the milk is less than half full, then buy new pack of milk
- always buy a small bag of chips (we eat much chips)
- if tomatos are less or equal to 3, then buy more
- if no cucumber, then buy more
- tomatoes and cucumbers should not be more than 5 each

To model this, we need first to recognize our variables and what type they should be.

- **milk_half**     : boolean   - since saying if it is half full (in fridge) is a yes or no question
- **milk_bought**   : boolean   - since saying if it is milk was bought is a yes or no question
- **chips**         : boolean   - just says bag of chips, no quantity
- **tomatoes**      : int       - the are restricted in a range between 4-5
- **cucumbers**     : int       - the are restricted in a range between 1-5

So let's set these variables up in code

.. code:: python

    import puan

    milk_home   = puan.variable("milk_home")
    milk_bought = puan.variable("milk_bought")
    chips       = puan.variable("chips")
    tomatoes    = puan.variable("tomatoes", int)
    cucumbers   = puan.variable("cucumbers", int)

Now we need to set up their logical relationships between one another. We start
off with the most easy one, being the **chips**. It only says *buy a bag of chips*. In other words,
**chips must always be true!** We do this by setting up a so called *boolean variable proposition*.

.. code:: python

    import puan.logic.cic as cc

    chips_is_true = cc.boolean_variable_proposition(chips)

Secondly we take a look at the **milk**. We were waying that if the milk in the fridge is 
half full, i.e milk_home is True, then it is implied that also the milk is bought, i.e milk_bought is True. 
This is called an implication:

.. math::

   \text{milk_done_right} = \text{milk_home} \rightarrow \text{milk_bought} 
   
There is a proposition class for this expression called `:class: puan.logic.cic.implication_proposition` and it
exist to model an implication. The implication-proposition is another essential class in the cic-system and it has three 
properties: an operator, a condition (optional), a consequence and a default. The operator says the relationship between 
the condition and the consequence. For instance, it can say *if milk is half* (condition), then *require* (operator ALL) 
to buy more milk (consequence). The default is used when the operator is ANY and it is saying which selecting is default. 

So now, let's put it into code.

.. code:: python

    milk_done_right = cc.implication_proposition(
        condition=cc.conditional_proposition("ALL", [
            cc.boolean_variable_proposition(milk_home)
        ]),
        implies=cc.Operator.ALL,
        consequence=cc.conditional_proposition("ALL", [
            cc.boolean_variable_proposition(milk_bought)
        ])
    )

And now for the last two chips ones. They are limited by some integers. Both the tomatoes and cucumbers must not be larger than 5.
And the tomatoes should not be less than 2. To model this we need to first create three `:class: puan.logic.cic.discrete_variable_proposition`

.. code:: python

    # tomatoes more or equal to 4
    tomatoes_ge_one = cc.discrete_variable_proposition(tomatoes, ">=", 4)
    
    # tomatoes and cucumbers less or equal to 5
    tomatoes_le_five = cc.discrete_variable_proposition(tomatoes, "<=", 5)
    cucumbers_le_five = cc.discrete_variable_proposition(cucumbers, "<=", 5)

    # cucumbers more or equal to 1 
    cucumbers_ge_one = cc.discrete_variable_proposition(cucumbers, ">=", 1)
    
Now if all of these variables are true, then it means that number of tomatoes are between 4-5 and number of cucumbers between 1-5.
To tie these two expressions we need to plug them into an implication proposition.

.. code:: python

    vegestables_ok = cc.implication_proposition(
        implies=cc.Operator.ALL,
        consequence=cc.conditional_proposition("ALL", [
            tomatoes_ge_one,
            tomatoes_le_five,
            cucumbers_le_five,
            cucumbers_ge_one
        ])
    )

Now we can put it all together in a single cic-model

.. code:: python

    fridge_model = cc.conjunctional_implication_proposition([
        chips_is_true,
        milk_done_right,
        vegestables_ok
    ])

And imagine now that we are going to the store and notice what we have in the fridge:

- milk is less than half full
- we have two tomatoes and no cucumbers

we go to the store and check our model with the current shopping cart after we added two tomatoes and one cucumber:

.. code:: python

    import puan.ndarray as pnd

    # we don't ever want to buy more than 10 of anything
    ph = fridge_model.to_polyhedron(integer_bounds=(0, 10)) 

    cart = ph.construct([(milk_home, 0), (milk_bought, 0), (tomatoes, 2+2), (cucumbers, 1)])
