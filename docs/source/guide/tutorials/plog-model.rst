Learn the PLog modelling system
===============================
In this tutorial we're going to learn about the Propositional Logic (PLog for short) modelling system 
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

    import puan.logic.plog as pg

    milk_home   = pg.Proposition(id="milk_home",   dtype=bool, virtual=False)
    milk_bought = pg.Proposition(id="milk_bought", dtype=bool, virtual=False)
    chips       = pg.Proposition(id="chips",       dtype=bool, virtual=False)
    tomatoes    = pg.Proposition(id="tomatoes",    dtype=int,  virtual=False)
    cucumbers   = pg.Proposition(id="cucumbers",   dtype=int,  virtual=False)

Now we need to set up their logical relationships between one another. We start by taking a look at the **milk**. 
We were waying that if the milk in the fridge is half full, i.e milk_home is True, then it is implied that also the milk is bought, i.e milk_bought is True. 
This is called an implication:

.. math::

   \text{milk_done_right} = \text{milk_home} \rightarrow \text{milk_bought} 
   
There is a proposition class for this expression called `:class: puan.logic.plog.Imply` and it exist to model an implication. 
The implication-proposition is another essential class in the plog system and it has two properties; a condition and a consequence. 
The operator says the relationship between the condition and the consequence. For instance, it can say *if milk is half* (condition), then buy more milk (consequence). 

So now, let's put it into code.

.. code:: python

    milk_done_right = pg.Imply(
        condition=Not(milk_home),
        consequence=milk_bought,
        id="milk_done_right"
    )

*Note that the `id` here is optional and not necessarily used.*

The tomatoes and cucumbers are limited by some integers. Both the number of tomatoes and cucumbers must not be larger than 5.
And the number of tomatoes should not be less than 2. To model this we do the following

.. code:: python

    # tomatoes more or equal to 4
    tomatoes_ge_four = pg.AtLeast(tomatoes, value=4, id="tomatoes_ge_four")

    # tomatoes and cucumbers less or equal to 5
    tomatoes_le_five = pg.AtMost(tomatoes, value=5, id="tomatoes_le_five")
    cucumbers_le_five = pg.AtMost(cucumbers, value=5, id="cucumbers_le_five")

    # cucumbers more or equal to 1 
    cucumbers_ge_one = pg.AtLeast(cucumbers, value=1, id="cucumbers_ge_one")
    
Now if all of these variables are true, then it means that number of tomatoes are between 4-5 and number of cucumbers between 1-5.
To tie these two expressions we need to plug them into an implication proposition.

.. code:: python

    vegestables_ok = pg.All(
        tomatoes_ge_four,
        tomatoes_le_five,
        cucumbers_le_five,
        cucumbers_ge_one,
        id="vegestables"
    )

Now we can put it all together in a single plog-model

.. code:: python

    fridge_model = pg.All(
        chips_is_true,
        milk_done_right,
        vegestables_ok,
        id="fridge"
    )

And imagine now that we are going to the store and notice what we have in the fridge:

- milk is less than half full
- we have two tomatoes and no cucumbers

we go to the store and check our model with the current shopping cart after we added two tomatoes and one cucumber:

.. code:: python

    # Convert fridge model to a polyhedron that we can use to calculate on
    ph = fridge_model.to_polyhedron(active=True)

    # Construct a cart numpy array instance from variables ...
    cart = ph.construct([
        (milk_home, 1), 
        (milk_bought, 0), 
        (tomatoes, 2+2), 
        (cucumbers, 0)
    ])

    # ... and evaluate if it satisfies the model
    print(ph.evaluate(cart))
    # >>> (False, integer_ndarray([1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 4]))

As expected, the current cart is not valid (we don't have *chips* nor *cucumbers*). Let's pick them from the store and
check again if we're now ok

.. code:: python

    # Construct a cart numpy array instance from variables ...
    new_cart = ph.construct([
        (chips,       1),
        (milk_home,   1), 
        (milk_bought, 0), 
        (tomatoes,    2+2), 
        (cucumbers,   1)
    ])

    # ... and evaluate if it satisfies the model
    print(ph.evaluate(new_cart))
    # >>> (True, integer_ndarray([1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 4]))

And now we are ready to checkout and go home.