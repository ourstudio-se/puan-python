Tutorial
========
In this tutorial we are going to create a part of a backend for a app called Wardrobe Wizard. The app is for creating 
outfits based on certain criterias. An outfit ends up in a vast number of combinations and are a perfect example of
illustrating how to use Puan.

.. image:: images/release-1.jpg
   :alt: alternate text
   :align: center

Creating an STA-model
---------------------

An **outfit** we say must have: 
    - exactly one pair of **shoes**
    - exactly one pair of **bottoms**
    - one or more **top** (s)
    - at most one **hat**

Now we also add some more specific rules to our model:
    - a **top** is either a **shirt** (exclusive) or a **t-shirt**
    - **bottoms** are either **shorts** (exclusive) or **trousers**

Now we would like to create an :ref:`STA -model <puan.logic.sta>`. First we create all category items:

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
            "id": "shirt",
            "name": "Shirt",
            "virtual": True,
            "category": {
                "id": "outfit"
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
        {
            "id": "t-shirt",
            "name": "T-Shrit",
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
            "id": "trousers",
            "name": "Trousers",
            "virtual": True,
            "category": {
                "id": "bottoms"
            }
        }
    ]


Notice the "virtual"-property on all these items. We do this to later keep control of what belongs as supporting variables in the model
and what items we actually care about later when doing different computations.

Now we create STA-rules that bound logic relations. The first rule we add will bound a requires-exclusively (xor) relation between an item and its 
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
                "value": "shirt"
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
                "value": "shirt"
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

Now we can compile into propositions and/or a polyhedron undependent on new items.

.. code:: python

    import puan.logic.sta as sta
    import puan.logic.cic as cc

    # We assume items come from some other source...
    # but hardcode some items here
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
    
    # Add together all items
    items = virtual_items + non_virtual_items
    sta_rules = [rule1, rule2]

    # Compile into a conjunctional proposition
    conj_prop = sta.application.to_conjunction_proposition(sta_rules, items)

    # Check if some combination is valid
    polyhedron = conj_prop.to_polyhedron()

    # Combination is not separable meaning it is inside the polyhedron
    assert not polyhedron.separable(
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

We check at the end if my outfit of Converse shoes, a pair of black trousers, a white t-shirt and a cool black hat is considered to be an
outfit in this model.

Finding combinations (with a solver)
------------------------------------

.. _npycvx: https://github.com/ourstudio-se/puan-npycvx

It is easy to check if a model satisfies a specific combination. But since the combination space tends to be very large, finding a specific one is hard. 
To find one in this context, we use a mixed integer linear programming solver and for this specific example we use `NpyCVX <npycvx>`.

Using the same model, we now want to try and find the outfit with as much clothes on as possible. 

.. code:: python

    import npycvx
    import puan.ndarray as pnd

    # We convert our polyhedron into cvxopt's constraints format 
    problem = npycvx.convert_numpy(*polyhedron.to_linalg())

    # Here we compute the search and tries to find an outfit with as much clothes as possible (maximizing positive one-vector)
    status, solution = npycvx.solve_lp(*problem, False, numpy.ones(len(polyhedron.A.variables)))

    if status == "optimal":

        # Print out the solution variables but skip the virtual ones 
        print(
            pnd.boolean_ndarray(
                solution, 
                polyhedron.A.variables
            ).to_list(True)
        )

        # [
        #   'black_hat_with_cool_label': <class 'bool'> , <- did you also read "class cool" 8) ?
        #   'black_trousers': <class 'bool'> , 
        #   'converse': <class 'bool'> , 
        #   'white_t_shirt': <class 'bool'> 
        # ]