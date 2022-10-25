.. _sta-model:

Learn the STA modelling system
==============================
In this tutorial we are going to learn the source-target application or STA modelling. After this tutorial you will know how to
define an STA model and deduce if it is satisfied given a combination of items.

The Wardrobe Wizard example
---------------------------
To demonstrate the STA modelling system we're going to create a part of a backend for an app called Wardrobe Wizard. The purpose of the app is to create 
outfits from several items based on certain criterias. With a reasonable amout of items the number of outfits ends up in a vast number of combinations and
therefore this is a perfect example of illustrating how to use Puan.

.. image:: images/release-1.jpg
   :alt: alternate text
   :align: center

Creating an STA model
---------------------

An **outfit** must have: 
    - exactly one pair of **shoes**
    - exactly one pair of **bottoms**
    - one or more **top** (s)
    - at most one **hat**

Now we add some more specific rules to our model:
    - a **top** is either a **shirt**, **blouse** or a **t-shirt** (exclusive, i.e. it cannot be both a blouse and a t-shirt)
    - **bottoms** are either **shorts**, **trousers** or **skirts** (exclusive) 

Based on those rules we will create an :ref:`STA model <puan.logic.sta>`. First we define all category items:

.. code:: python

    virtual_items = [
        {
            "id": "outfit",
            "name": "Outfit",
            "virtual": True
        },
        {
            "id": "shoes",
            "name": "Shoes",
            "virtual": True,
            "category": {
                "id": "outfit"
            }
        },
        {
            "id": "bottoms",
            "name": "Bottoms",
            "virtual": True,
            "category": {
                "id": "outfit"
            }
        },
        {
            "id": "shorts",
            "name": "Shorts",
            "virtual": True,
            "category": {
                "id": "bottoms"
            }
        },
        {
            "id": "skirts",
            "name": "Skirts",
            "virtual": True,
            "category": {
                "id": "bottoms"
            }
        },
        {
            "id": "trousers",
            "name": "Trousers",
            "virtual": True,
            "category": {
                "id": "bottoms"
            }
        },
        {
            "id": "top",
            "name": "Top",
            "virtual": True,
            "category": {
                "id": "outfit"
            }
        },
        {
            "id": "shirt",
            "name": "Shirt",
            "virtual": True,
            "category": {
                "id": "top"
            }
        },
        {
            "id": "blouse",
            "name": "Blouse",
            "virtual": True,
            "category": {
                "id": "top"
            }
        },
        {
            "id": "t-shirt",
            "name": "T-Shrit",
            "virtual": True,
            "category": {
                "id": "top"
            }
        },
        {
            "id": "hat",
            "name": "Hat",
            "virtual": True,
            "category": {
                "id": "outfit"
            }
        },
    ]


Notice the "virtual"-property on all these items. We have this to keep track of the so called supporting variables in the model.
Typically, supporting variables are not of interest to the end-user, e.g. in this example the end-user is not interested in knowing that **top** is selected but rather which particular top was.

Next we create STA rules for logic relations between the variables in the model. The first rule we add will bound a requires-exclusively (xor) relation between an item and its 
category item. In other words it says, if the category is selected then select exactly one of the items having that category.

.. code:: python

    rule1 = {
        "variables": [
            {
                "key": "variable",
                "value": "outfit"
            },
            {
                "key": "variable",
                "value": "bottoms"
            },
            {
                "key": "variable",
                "value": "shoes"
            },
            {
                "key": "variable",
                "value": "trousers"
            },
            {
                "key": "variable",
                "value": "shorts"
            },
            {
                "key": "variable",
                "value": "skirts"
            },
            {
                "key": "variable",
                "value": "shirt"
            },
            {
                "key": "variable",
                "value": "blouse"
            },
            {
                "key": "variable",
                "value": "t-shirt"
            },
            {
                "key": "variable",
                "value": "hat"
            }
        ],
        "source": {
            "selector": {
                "active": True,
                "conjunctionSelector": {
                    "disjunctions": [
                        {
                            "literals": [
                                {
                                    "key": "id",
                                    "operator": "==",
                                    "value": "$variable"
                                }
                            ]
                        }
                    ]
                }
            }
        },
        "target": {
            "selector": {
                "active": True,
                "conjunctionSelector": {
                    "disjunctions": [
                        {
                            "literals": [
                                {
                                    "key": "category.id",
                                    "operator": "==",
                                    "value": "$variable"
                                }
                            ]
                        }
                    ]
                }
            }
        },
        "apply": {
            "ruleType": "REQUIRES_EXCLUSIVELY"
        }
    }

The second rule binds back from the items to their category item.

.. code:: python

    rule2 = {
        "variables": [
            {
                "key": "variable",
                "value": "bottom"
            },
            {
                "key": "variable",
                "value": "shoes"
            },
            {
                "key": "variable",
                "value": "trousers"
            },
            {
                "key": "variable",
                "value": "short"
            },
            {
                "key": "variable",
                "value": "skirt"
            },
            {
                "key": "variable",
                "value": "shirt"
            },
            {
                "key": "variable",
                "value": "blouse"
            },
            {
                "key": "variable",
                "value": "t-shirt"
            },
            {
                "key": "variable",
                "value": "top"
            },
            {
                "key": "variable",
                "value": "hat"
            }
        ],
        "source": {
            "groupBy": {
                "onKey": "category.id"
            },
            "selector": {
                "active": True,
                "conjunctionSelector": {
                    "disjunctions": [
                        {
                            "literals": [
                                {
                                    "key": "category.id",
                                    "operator": "==",
                                    "value": "$variable"
                                }
                            ]
                        }
                    ]
                }
            }
        },
        "target": {
            "selector": {
                "active": True,
                "conjunctionSelector": {
                    "disjunctions": [
                        {
                            "literals": [
                                {
                                    "key": "id",
                                    "operator": "==",
                                    "value": "$variable"
                                }
                            ]
                        }
                    ]
                }
            }
        },
        "apply": {
            "ruleType": "REQUIRES_ALL",
            "conditionRelation": "ANY"
        }
    }

Given those rules it is possible to compile into propositions and/or a polyhedron which is used to compute valid combinations.
Yet the model is independent on items, which makes it very convenient to update the supply of items.

.. code:: python

    import puan.logic.sta as sta

    # In the real application items come from some other source...
    # but for this example we hardcode a subset of the items here
    non_virtual_items = [
        {
            "id": "black_trousers",
            "name": "Black trousers",
            "category": {
                "id": "trousers"
            }
        },
        {
            "id": "blue_trousers",
            "name": "Blue trousers",
            "category": {
                "id": "trousers"
            }
        },
        {
            "id": "white_t_shirt",
            "name": "White T-Shirt",
            "category": {
                "id": "t-shirt"
            }
        },
        {
            "id": "blue_t_shirt",
            "name": "Blue T-Shirt",
            "category": {
                "id": "t-shirt"
            }
        },
        {
            "id": "green_t_shirt",
            "name": "Green T-Shirt",
            "category": {
                "id": "t-shirt"
            }
        },
        {
            "id": "converse",
            "name": "Converse",
            "category": {
                "id": "shoes"
            }
        },
        {
            "id": "black_hat_with_cool_label",
            "name": "Black Hat with Cool Label",
            "category": {
                "id": "hat"
            }
        },
    ]
    
    # Join the virtual and non-virtual items
    items = virtual_items + non_virtual_items
    sta_rules = [rule1, rule2]

    # Compile into an `All` proposition
    conj_prop = sta.application.to_all_proposition(sta_rules, items)

    # Check if a particular combination is valid
    polyhedron = conj_prop.to_polyhedron()
    assert polyhedron.ineqs_satisfied(
        polyhedron.construct_boolean_ndarray([
            "converse",
            "black_trousers",
            "white_t_shirt",
            "black_hat_with_cool_label",
            "hat",
            "bottoms",
            "outfit",
            "top",
            "t-shirt",
            "trousers",
            "shoes",
        ])
    )

In the last step we check if my outfit of Converse shoes, a pair of black trousers, a white t-shirt and a cool black hat is considered to be a valid
outfit in this model.

Finding specific solution (with a solver)
-----------------------------------------

.. _npycvx: https://github.com/ourstudio-se/puan-npycvx

It is easy to check if a given combination is satisfied for a given model. But finding a specific combination is hard since the combination space tends to be very large.
Puan converts the logic model into a mixed integer linear program to which a solver can be applied in order to find specific combinations.  
For this specific example we use the `NpyCVX <npycvx>` solver.

Using the same model as previously defined, we now want to find the outfit with as much clothes as possible. 

.. code:: python

    import npycvx
    import numpy as np
    import puan.ndarray as pnd

    # We convert our polyhedron into cvxopt's constraints format
    problem = npycvx.convert_numpy(*polyhedron.to_linalg())

    # Here we solve the linear program to find an outfit with as much clothes as possible (maximizing positive one-vector)
    status, solution = npycvx.solve_lp(*problem, False, np.ones(len(polyhedron.A.variables)))

    if status == "optimal":

        # Print out the solution variables
        print(
            pnd.boolean_ndarray(
                solution,
                polyhedron.A.variables
            ).to_list(True)
        )