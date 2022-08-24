import ast
import functools
import puan
import puan.misc
import puan.ndarray
import puan.logic
import puan.logic.logicfunc as lf
import puan.logic.plog as pg
import puan.modules.configurator as cc
import numpy
import operator
import maz

from hypothesis import example, given, strategies as st, settings, assume

def atom_proposition_strategy():
    return st.builds(
        pg.Proposition,
        id=st.text(), 
        dtype=st.binary(min_size=1, max_size=1),
        virtual=st.binary(min_size=1, max_size=1),
        value=st.integers(min_value=-5, max_value=5),
        sign=st.binary(min_size=1, max_size=1),
    )

def proposition_strategy():

    def extend(prop):
        return st.builds(
            pg.Proposition, 
            prop,
            id=st.text(), 
            dtype=st.binary(min_size=1, max_size=1),
            virtual=st.binary(min_size=1, max_size=1),
            value=st.integers(min_value=-5, max_value=5),
            sign=st.binary(min_size=1, max_size=1),
        )

    return st.builds(
        pg.Proposition, 
        st.iterables(
            st.recursive(
                atom_proposition_strategy(), 
                extend
            ),
            max_size=5
        ),
        id=st.text(), 
        dtype=st.integers(min_value=0, max_value=1),
        virtual=st.booleans(),
        value=st.integers(min_value=-5, max_value=5),
        sign=st.sampled_from([-1,1]),
    )

@given(proposition_strategy(), st.integers(min_value=-1, max_value=1))
def test_model_assume_to_polyhedron_hypothesis(model, value):
    model.assume({model.variables[0].id: value})[0].to_polyhedron()

@settings(deadline=None)
@given(proposition_strategy(), st.integers(min_value=-1, max_value=1), st.integers(min_value=0, max_value=10))
def test_model_assume_interpretation_hypothesis(model, value, n_vars):
    
    """
        We test that if an interpretation of model is in the assumed model,
        then that implies that it is also in the original model. 
    """

    # If the model is contradiction, then we doesn't need to test (also it will not work)
    if not model.is_contradiction:
        selected_variable = numpy.random.choice(model.variables)
        assumed, consequence = model.assume({v.id: value for v in numpy.random.choice(model.variables, size=n_vars)})
        
        original_polyhedron = model.to_polyhedron(True)
        assumed_polyhedron = assumed.to_polyhedron()

        random_interpretation = {v.id: 1 for v in numpy.random.choice(assumed.variables, size=n_vars)}
        original_interpretation_vector = original_polyhedron.construct(*random_interpretation.items())
        assumed_interpretation_vector = assumed_polyhedron.construct(*random_interpretation.items())

        # Now, if interpretation is in the assumed model, then it impies that is should be in original model as well
        # (Implication x -> y means -x v y)
        assert (not all(assumed_polyhedron.A.dot(assumed_interpretation_vector) >= assumed_polyhedron.b)) or all(original_polyhedron.A.dot(original_interpretation_vector) >= original_polyhedron.b)

    
@given(proposition_strategy())
def test_model_variables_hypothesis(model):
    _ = model.variables

@given(proposition_strategy())
def test_specity_proposition_hypothesis(model):
    _ = model.specify()


def test_application_to_rules():
    items = [
        {"id": 0, "type": "M"},
        {"id": 1, "type": "M"},
        {"id": 2, "type": "M"},
        {"id": 3, "type": "M"},
        {"id": 4, "type": "N"},
        {"id": 5, "type": "N"},
        {"id": 6, "type": "N"},
        {"id": 7, "type": "O"},
        {"id": 8, "type": "O"},
    ]
    rules_gen = puan.logic.sta.application.to_plog(
        {
            "source": {
                "selector": {
                    "active":True,
                    "conjunctionSelector": {
                        "disjunctions":[
                            {
                                "literals":[
                                    {
                                        "key":"id",
                                        "operator":">",
                                        "value":3,
                                    }
                                ],
                            }
                        ]
                    }
                }
            },
            "target": {
                "selector": {
                    "active":True,
                    "conjunctionSelector": {
                        "disjunctions":[
                            {
                                "literals":[
                                    {
                                        "key":"id",
                                        "operator":"<=",
                                        "value":3
                                    }
                                ]
                            }
                        ]
                    }
                }
            },
            "apply": {
                "ruleType":"REQUIRES_ANY",
                "conditionRelation": "ALL"
            },
        },
        from_items=items,
    )
    rules_list = list(rules_gen)
    assert len(rules_list) == 1
    assert len(rules_list[0]["condition"]["subConditions"][0]["components"]) == 5
    assert len(rules_list[0]["consequence"]["components"]) == 4

def test_rules2variables():

    rules = [
        {
            "id": "",
            "condition": {},
            "consequence": {
                "ruleType": "REQUIRES_ALL",
                "components": [
                    {"code": "x"},
                    {"code": "y"}
                ]
            }
        },
        {
            "id": "",
            "condition": {
                "relation": "ALL",
                "subConditions": [
                    {
                        "relation": "ANY",
                        "components": [
                            {"code": "a"}
                        ]
                    },
                    {
                        "relation": "ALL",
                        "components": [
                            {"code": "b"},
                            {"code": "c"}
                        ]
                    }
                ]
            },
            "consequence": {
                "ruleType": "REQUIRES_ALL",
                "components": [
                    {"code": "x"},
                    {"code": "y"}
                ]
            }
        },
        {
            "id": "",
            "consequence": {
                "ruleType": "REQUIRES_ALL",
                "components": [
                    {"code": "z"}
                ]
            }
        },
    ]

    model = pg.All(*map(functools.partial(pg.Imply.from_cicJE, id_ident="code"), rules))
    variables = list(map(operator.attrgetter("id"), model.variables_full()))
    assert set(["a", "b", "c", "x", "y", "z"]).issubset(set(variables))

def test_rules2matrix_with_mixed_condition_rules():

    """
        Rules with mixed condition (has at least one any-relation),
        has a different parser than non-mixed. We test it here.
    """
    rules = [
        {
            "id": "B",
            "condition": {
                "relation": "ANY",
                "subConditions": [
                    {
                        "relation": "ALL",
                        "components": [
                            {"id": "a"},
                            {"id": "b"}
                        ],
                        "id":"E"
                    },
                    {
                        "relation": "ALL",
                        "components": [
                            {"id":"c"},
                            {"id":"d"}
                        ],
                        "id":"F"
                    }
                ],
                "id": "D"
            },
            "consequence": {
                "ruleType": "REQUIRES_ALL",
                "components": [
                    {"id": "x"},
                    {"id": "y"},
                ],
                "id": "C"
            }
        }
    ]
    conj_props = pg.All(*map(pg.Imply.from_cicJE, rules), id="A")
    matrix = conj_props.to_polyhedron(active=True)

    expected_feasible_configurations = matrix.construct(
        [("B",1),("D",1),("E",1),("F",1)],
        [("B",1),("C",1),("x",1),("y",1)],
        [("B",1),("C",1),("a",1),("b",1),("x",1),("y",1)],
        [("B",1),("D",1),("E",1),("F",1),("a",1),("c",1)]
    )
    expected_infeasible_configurations = matrix.construct(
       #"a  b  c  d  x  y"
       [],
       [("B",1),("C",1),("E",1),("F",1),("a",1),("b",1)],
       [("B",1),("C",1),("E",1),("F",1),("c",1),("d",1)]
    )

    eval_fn = maz.compose(all, functools.partial(operator.le, matrix.b), matrix.A.dot)
    assert all(map(eval_fn, expected_feasible_configurations))
    assert not any(map(eval_fn, expected_infeasible_configurations))

def test_extract_value_from_list_when_index_out_of_range():
    try:
        puancore.extract_value(
            from_dict={
                "m": [
                    {
                        "k": 1,
                    }
                ]
            },
            selector_string="m.1.k"
        )
        assert False
    except:
        pass

def test_extract_value_from_list_value():
    value = puan.logic.sta.application._extract_value(
        from_dict={
            'l': [
                {
                    "value": 10
                }
            ]
        },
        selector_string="l.0.value"
    )
    assert value == 10

def test_extract_value_from_list_with_lt():
    value = puan.logic.sta.application._extract_value(
        from_dict={
            'l': [
                {
                    "value": 10
                },
                {
                    "value": 8
                },
                {
                    "value": 9
                },
            ]
        },
        selector_string="l.<.value"
    )
    assert value == 10

def test_extract_value_from_list_with_gt():
    value = puan.logic.sta.application._extract_value(
        from_dict={
            'l': [
                {
                    "value": 10
                },
                {
                    "value": 8
                },
                {
                    "value": 9
                },
            ]
        },
        selector_string="l.>.value"
    )
    assert value == 9

def test_extract_value_simple():
    value = puan.logic.sta.application._extract_value(
        from_dict={
            "k": {
                "i": 0
            }
        },
        selector_string="k.i"
    )
    assert value == 0

def test_validate_item_from_assert_true():
    assert puan.logic.sta.application._validate_item_from(
        {
            "disjunctions":[
                {
                    "literals":[
                        {
                            "key":"m.k",
                            "operator":"==",
                            "value":3,
                        }
                    ]
                }
            ]
        },
        {
            "m": {
                "k": 3
            }
        }
    )

def test_validate_item_from_assert_false():
    assert not puan.logic.sta.application._validate_item_from(
        {
            "disjunctions":[
                {
                    "literals":[
                        {
                            "key":"m.0.k",
                            "operator":"==",
                            "value":4,
                        }
                    ]
                }
            ]
        },
        {
            "m": [
                {
                    "k": 3
                }
            ]
        }
    )

def test_extract_items_from():
    items = [
        {
            "m": [
                {
                    "k": {},
                }
            ],
            "n": "ney"
        },
        {
            "m": [
                {
                    "k": "mklasd",
                }
            ],
            "n": "yey"
        },
        {
            "m": [
                {
                    "k": 0,
                }
            ],
            "n": "yey"
        },
        {
            "m": [
                {
                    "k": 1,
                }
            ],
            "n": "yey"
        },
        {
            "m": [
                {
                    "k": 2,
                }
            ],
            "n": "yey"
        },
        {
            "m": [
                {
                    "k": 3,
                }
            ],
            "n": "yey"
        },
    ]
    extracted_items = puan.logic.sta.application._extract_items_from(
        {
            "disjunctions":[
                {
                    "literals":[
                        {
                            "key":"m.0.k",
                            "operator":"==",
                            "value":0,
                        },
                        {
                            "key":"m.0.k",
                            "operator":"==",
                            "value":1,
                        }
                    ],
                },
                {
                    "literals":[
                        {
                            "key":"n",
                            "operator":"==",
                            "value":"yey",
                        },
                    ],
                }
            ]
        },
        items,
    )

    assert list(extracted_items) == items[2:4]

def test_apply_selector_should_fail():
    items = [
        {
            "m": [
                {
                    "k": {},
                }
            ],
            "n": "ney"
        },
        {
            "m": [
                {
                    "k": "mklasd",
                }
            ],
            "n": "yey"
        },
        {
            "m": [
                {
                    "k": 0,
                }
            ],
            "n": "yey"
        },
        {
            "m": [
                {
                    "k": 1,
                }
            ],
            "n": "yey"
        },
        {
            "m": [
                {
                    "k": 2,
                }
            ],
            "n": "yey"
        },
        {
            "m": [
                {
                    "k": 3,
                }
            ],
            "n": "yey"
        },
    ]

    try:
        puan.logic.sta.application._apply_selector(
            selector={
                "active":True,
                "conjunctionSelector": {
                    "disjunctions":[
                        {
                            "literals":[
                                {
                                    "key":"m.0.k",
                                    "operator":"==",
                                    "value":0,
                                },
                                {
                                    "key":"m.0.k",
                                    "operator":"==",
                                    "value":1,
                                },
                            ],
                        },
                        {
                            "literals":[
                                {
                                    "key":"n",
                                    "operator":"==",
                                    "value":"yey",
                                },
                            ],
                        },
                    ]
                },
                "requirements":["EXACTLY_ONE"]
            },
            to_items=items,
        )
        assert False
    except:
        pass

def test_apply_selector_should_pass():
    items = [
        {
            "m": [
                {
                    "k": {},
                }
            ],
            "n": "ney"
        },
        {
            "m": [
                {
                    "k": "mklasd",
                }
            ],
            "n": "yey"
        },
        {
            "m": [
                {
                    "k": 0,
                }
            ],
            "n": "yey"
        },
        {
            "m": [
                {
                    "k": 2,
                }
            ],
            "n": "yey"
        },
        {
            "m": [
                {
                    "k": 3,
                }
            ],
            "n": "yey"
        },
    ]

    extracted_items = puan.logic.sta.application._apply_selector(
        selector={
            "active":True,
            "conjunctionSelector": {
                "disjunctions": [
                    {
                        "literals":[
                            {
                                "key":"m.0.k",
                                "operator":"==",
                                "value":0,
                            },
                            {
                                "key":"m.0.k",
                                "operator":"==",
                                "value":1,
                            }
                        ]
                    },
                    {
                        "literals":[
                            {
                                "key":"n",
                                "operator":"==",
                                "value":"yey",
                            },
                        ],
                    },
                ]
            },
            "requirements":[
                "EXACTLY_ONE"
            ]
        },
        to_items=items,
    )
    assert extracted_items == items[2:3]

def test_apply_collector():
    collection = puan.logic.sta.application._apply_collector(
        collector={
            "groupBy": {
                "onKey":"type",
            },
            "selector": {
                "active":True,
            }
        },
        to_items=[
            {"id": 0, "type": "M"},
            {"id": 1, "type": "M"},
            {"id": 2, "type": "M"},
            {"id": 3, "type": "M"},
            {"id": 4, "type": "N"},
            {"id": 5, "type": "N"},
            {"id": 6, "type": "N"},
            {"id": 7, "type": "O"},
            {"id": 8, "type": "O"},
        ]
    )
    assert len(collection) == 3
    assert len(collection[0]) == 4
    assert len(collection[1]) == 3
    assert len(collection[2]) == 2

def test_application_to_rules():
    items = [
        {"id": "a", "type": 1},
        {"id": "b", "type": 2},
        {"id": "c", "type": 3},
        {"id": "d", "type": 4},
        {"id": "e", "type": 5},
        {"id": "f", "type": 6},
        {"id": "g", "type": 6},
        {"id": "h", "type": 6},
        {"id": "i", "type": 7},
    ]
    model = puan.logic.sta.application.to_plog(
        {
            "source": {
                "selector": {
                    "active":True,
                    "conjunctionSelector": {
                        "disjunctions":[
                            {
                                "literals":[
                                    {
                                        "key":"type",
                                        "operator":">",
                                        "value":3,
                                    }
                                ],
                            }
                        ]
                    }
                }
            },
            "target": {
                "selector": {
                    "active":True,
                    "conjunctionSelector": {
                        "disjunctions":[
                            {
                                "literals":[
                                    {
                                        "key":"type",
                                        "operator":"<=",
                                        "value":3
                                    }
                                ]
                            }
                        ]
                    }
                }
            },
            "apply": {
                "ruleType":"REQUIRES_ANY",
                "conditionRelation": "ALL"
            },
        },
        from_items=items,
        model_id="test"
    )
    assert len(model.propositions) == 1
    first_prop = model.propositions[0]
    assert isinstance(first_prop.propositions[0], pg.AtLeast)
    assert len(first_prop.propositions[0].propositions) == 3
    assert isinstance(first_prop.propositions[1], pg.AtMost)
    assert len(first_prop.propositions[1].propositions) == 6

def test_application_to_rules_with_condition_relation():
    items = [
        {"id": 0, "type": "M"},
        {"id": 1, "type": "M"},
        {"id": 2, "type": "M"},
        {"id": 3, "type": "M"},
        {"id": 4, "type": "N"},
        {"id": 5, "type": "N"},
        {"id": 6, "type": "N"},
        {"id": 7, "type": "O"},
        {"id": 8, "type": "O"},
    ]
    model = puan.logic.sta.application.to_plog(
        {
            "source": {
                "groupBy": {
                    "onKey":"type",
                },
                "selector": {
                    "active":True,
                    "conjunctionSelector": {
                        "disjunctions":[
                            {
                                "literals":[
                                    {
                                        "key":"id",
                                        "operator":">",
                                        "value":3,
                                    }
                                ],
                            }
                        ]
                    }
                }
            },
            "target": {
                "selector": {
                    "active":True,
                    "conjunctionSelector": {
                        "disjunctions":[
                            {
                                "literals":[
                                    {
                                        "key":"id",
                                        "operator":"<=",
                                        "value":3
                                    }
                                ]
                            }
                        ]
                    }
                }
            },
            "apply": {
                "ruleType":"REQUIRES_ANY",
                "conditionRelation": "ANY",
            },
        },
        from_items=items,
    )
    assert len(model.propositions) == 2
    assert len(model.propositions[0].propositions) == 2
    assert len(model.propositions[1].propositions) == 2

def test_extract_value_with_key_missing():
    items = [
        {
            "id": 0,
            "name": "Rikard",
            "age": 31,
        },
        {
            "id": 0,
            "name": "Sara",
            "age": 26,
        },
        {
            "id": 0,
            "name": "Lisa",
        },
    ]

    applied_items = puan.logic.sta.application._apply_selector(
        selector={
            "active":True,
            "conjunctionSelector": {
                "disjunctions":[
                    {
                        "literals":[
                            {
                                "key":"age",
                                "operator":">",
                                "value":25,
                                "skipIfKeyError": True,
                            }
                        ]
                    }
                ]
            }
        },
        to_items=items,
    )
    assert len(applied_items) == 2

def test_apply_selector_will_return_no_items():
    items = [
        {"id": 0, "type": "M"},
        {"id": 1, "type": "M"},
        {"id": 2, "type": "M"},
        {"id": 3, "type": "M"},
        {"id": 4, "type": "N"},
        {"id": 5, "type": "N"},
        {"id": 6, "type": "N"},
        {"id": 7, "type": "O"},
        {"id": 8, "type": "O"},
    ]
    applied_items = puan.logic.sta.application._apply_selector(
        selector={
            "active":False,
            "conjunctionSelector": {},
        },
        to_items=items
    )
    assert len(applied_items) == 0

def test_apply_selector_will_return_all_items():
    items = [
        {"id": 0, "type": "M"},
        {"id": 1, "type": "M"},
        {"id": 2, "type": "M"},
        {"id": 3, "type": "M"},
        {"id": 4, "type": "N"},
        {"id": 5, "type": "N"},
        {"id": 6, "type": "N"},
        {"id": 7, "type": "O"},
        {"id": 8, "type": "O"},
    ]
    applied_items = puan.logic.sta.application._apply_selector(
        selector={
            "active":True,
            "conjunctionSelector": {},
        },
        to_items=items
    )
    assert len(applied_items) == len(items)

def test_application_with_no_item_hits_should_yield_no_rules():

    """
        When there are no variable hits from neither source or targets,
        then there should be no rules created. There was a risk before of
        creating a rule with no condition nor consequence.components.
    """

    application = {
        "variables": [
            {
                "key": "variable",
                "value": "c"
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
    items = [
        {"category": {"id": "X"}, "id": "a"},
        {"category": {"id": "X"}, "id": "b"},
        {"category": {"id": "Y"}, "id": "a"},
        {"category": {"id": "Y"}, "id": "b"},
    ]
    model = puan.logic.sta.application.to_plog(application, items)
    assert len(model.propositions) == 0


def test_reduce_matrix():
    
    matrix = puan.ndarray.ge_polyhedron(numpy.array([
        [-1,-1,-1, 1, 0, 2], # stay
        [-2, 0, 0, 0, 0, 0], # remove
        [ 0,-1, 1, 1, 0, 0], # stay
        [-3,-2,-1,-1, 0, 0], # stay
        [-4,-1,-1,-1,-1,-1], # stay
        [ 2, 1, 1, 1, 1, 1], # stay
        [ 0, 0, 0, 0, 0, 0], # remove
        [ 0, 1, 1, 0, 1,-1], # stay
    ]))
    reducable_rows = puan.ndarray.reducable_rows(matrix)
    actual = puan.ndarray.reduce(matrix, rows_vector=reducable_rows)
    expected = numpy.array([
        [-1,-1,-1, 1, 0, 2], # stay
        [ 0,-1, 1, 1, 0, 0], # stay
        [-3,-2,-1,-1, 0, 0], # stay
        [-4,-1,-1,-1,-1,-1], # stay
        [ 2, 1, 1, 1, 1, 1], # stay
        [ 0, 1, 1, 0, 1,-1], # stay
    ])
    assert numpy.array_equal(actual,expected)


def test_reducable_columns_approx():
    input = puan.ndarray.ge_polyhedron(numpy.array([[0, -1, -1, -1]]))
    actual = input.reducable_columns_approx()
    expected = numpy.array([0, 0, 0])
    assert numpy.array_equal(actual, expected)
    input = puan.ndarray.ge_polyhedron(numpy.array([[3, 1, 1, 1]]))
    actual = input.reducable_columns_approx()
    expected = numpy.array([1, 1, 1])
    assert numpy.array_equal(actual, expected)
    input = puan.ndarray.ge_polyhedron(numpy.array([[0, 1, 1, -3]]))
    actual = input.reducable_columns_approx()
    expected = numpy.array([numpy.nan, numpy.nan, 0])
    assert numpy.array_equal(actual, expected, equal_nan=True)
    input = puan.ndarray.ge_polyhedron(numpy.array([[2, 1, 1, -1]]))
    actual = input.reducable_columns_approx()
    expected = numpy.array([1, 1, 0])
    assert numpy.array_equal(actual, expected)
    input = puan.ndarray.ge_polyhedron(numpy.array([
        [ 0,-1, 1, 0, 0, 0], # 1
        [ 0, 0,-1, 1, 0, 0], # 2
        [-1,-1, 0,-1, 0, 0], # 3 1+2+3 -> Force not variable 0
    ]))
    actual = input.reducable_columns_approx()
    expected = numpy.array([ 0, 0, 0, 0, 0])*numpy.nan
    assert numpy.array_equal(actual, expected, equal_nan=True)
    input = puan.ndarray.ge_polyhedron(numpy.array([
        [1, 1],
        [1, -1]
    ]))
    actual = input.reducable_columns_approx()
    expected = numpy.array([numpy.nan])
    assert numpy.array_equal(actual, expected, equal_nan=True)
    input = puan.ndarray.ge_polyhedron(numpy.array([
        [0,-1,-1, 0, 0],
        [0, 0, 0,-1,-1]]),
        [
            puan.variable("0", 1, True),
            puan.variable("a", 0, False),
            puan.variable("b", 0, False),
            puan.variable("c", 0, False),
            puan.variable("d", 0, False)
        ])
    actual = input.reducable_columns_approx()
    expected = numpy.array([0,0,0,0])
    assert numpy.array_equal(actual, expected, equal_nan=True)




# def test_split_ruleset():
#     rules = [
#         {'condition': {
#             'relation': "ALL",
#             'subConditions': [{
#                 'relation': "ALL",
#                 'components': [
#                     {'id': 'a'},
#                     {'id': 'b'}
#                     ]
#                 }]
#             },
#         'consequence': {
#             'ruleType': 'REQUIRES_ANY',
#             'components': [
#                 {'id': 'j'},
#                 {'id': 'k'}
#                 ]
#             }
#         },
#         {'condition': {
#             'relation': "ALL",
#             'subConditions': [{
#                 'relation': "ANY",
#                 'components': [
#                     {'id': 'b'},
#                     {'id': 'c'}
#                     ]
#                 }]
#             },
#         'consequence': {
#             'ruleType': 'REQUIRES_ALL',
#             'components': [
#                 {'id': 'l'},
#                 {'id': 'm'}
#                 ]
#             }
#         },
#         {'condition': {
#             'relation': "ALL",
#             'subConditions': []
#             },
#         'consequence': {
#             'ruleType': 'REQUIRES_ALL',
#             'components': [
#                 {'id': 'a'},
#                 {'id': 'n'}
#                 ]
#             }
#         },
#         {'condition': {
#             'relation': "ALL",
#             'subConditions': []
#             },
#         'consequence': {
#             'ruleType': 'REQUIRES_ALL',
#             'components': [
#                 {'id': 'o'},
#                 {'id': 'p'}
#                 ]
#             }
#         },
#         {'condition': {
#             'relation': "ALL",
#             'subConditions': [{
#                 'relation': "ALL",
#                 'components': [
#                     {'id': 'p'},
#                     ]
#                 }]
#             },
#         'consequence': {
#             'ruleType': 'REQUIRES_ALL',
#             'components': [
#                 {'id': 'q'},
#                 ]
#             }
#         }
#     ]
#     expected_result = [
#         [
#             {'condition': {'relation': 'ALL', 'subConditions': [{'relation': 'ALL', 'components': [{'id': 'a'}, {'id': 'b'}]}]}, 'consequence': {'ruleType': 'REQUIRES_ANY', 'components': [{'id': 'j'}, {'id': 'k'}]}},
#             {'condition': {'relation': 'ALL', 'subConditions': []}, 'consequence': {'ruleType': 'REQUIRES_ALL', 'components': [{'id': 'a'}, {'id': 'n'}]}},
#             {'condition': {'relation': 'ALL', 'subConditions': [{'relation': 'ANY', 'components': [{'id': 'b'}, {'id': 'c'}]}]}, 'consequence': {'ruleType': 'REQUIRES_ALL', 'components': [{'id': 'l'}, {'id': 'm'}]}}
#         ],
#         [
#             {'condition': {'relation': 'ALL', 'subConditions': []}, 'consequence': {'ruleType': 'REQUIRES_ALL', 'components': [{'id': 'o'}, {'id': 'p'}]}},
#             {'condition': {'relation': 'ALL', 'subConditions': [{'relation': 'ALL', 'components': [{'id': 'p'}]}]}, 'consequence': {'ruleType': 'REQUIRES_ALL', 'components': [{'id': 'q'}]}}
#         ]
#     ]
#     split_ruleset = puan.logic.cic.cicJEs.split(rules)
#     #assert split_ruleset == expected_result

def test_cicJE_to_implication_proposition():

    actual_output = pg.Imply.from_cicJE({
        "condition": {
            "relation": "ALL",
            "subConditions": [
                {
                    "relation": "ANY",
                    "components": [
                        {"id": "x"},
                        {"id": "y"}
                    ]
                },
                {
                    "relation": "ANY",
                    "components": [
                        {"id": "a"},
                        {"id": "b"}
                    ]
                }
            ]
        },
        "consequence": {
            "ruleType": "REQUIRES_ALL",
            "components": [
                {"id": "m"},
                {"id": "n"},
                {"id": "o"},
            ]
        }
    })

    expected_output = pg.Imply(
        pg.All(
            pg.Any(*"xy"),
            pg.Any(*"ab"),
        ),
        pg.All(*"mno")
    )
    assert actual_output == expected_output

def test_neglect_columns():
    inputs = (
        numpy.array(  # M
            [
                    [0,-1, 1, 0, 0],
                    [0, 0,-1, 1, 0],
                    [0, 0, 0,-1, 1],
            ]),
        numpy.array(  # columns_vector
                    [1, 0, 1, 0]
            )
    )
    actual = puan.ndarray.neglect_columns(*inputs)
    expected = numpy.array([
                    [ 1, 0, 1, 0, 0],
                    [-1, 0,-1, 0, 0],
                    [ 1, 0, 0, 0, 1],
                ])
    assert numpy.array_equal(actual, expected)

def test_configuration2value_map():
    inputs = (
        [
            ('a', 'b', 'c'),
            ('c', 'd', 'e'),
        ],
        ['a', 'b', 'c', 'd', 'e']
    )
    actual = puan.ndarray.boolean_ndarray.from_list(*inputs).to_value_map()
    expected = {
        1: [
            [0, 0, 0, 1, 1, 1],
            [0, 1, 2, 2, 3, 4]
        ]
    }
    assert actual == expected


def test_reducable_matrix_columns_should_keep_zero_columns():

    """
        When reducing columns, we should not say anything
        about "zero"-columns.
    """

    M = puan.ndarray.ge_polyhedron([
        [ 1, 0, 1, 0, 0],
        [-3,-2,-1,-1, 0],
    ]).astype(numpy.int32)

    rows, cols = M.reducable_rows_and_columns()
    assert numpy.isnan(cols[3])
    assert cols[1] == 1


def test_ndint_compress():
    test_cases = [
        (
            puan.ndarray.integer_ndarray([
                [
                    [-4, 1, 2,-4,-4,-4],
                    [ 0, 0, 0, 1, 0, 0],
                ],
            ]),
            puan.ndarray.integer_ndarray([
                [-4, 1, 2, 16,-4,-4]
            ]),
            0
        ),
        (
            puan.ndarray.integer_ndarray([
                [
                    [ 0, 0, 0, 0, 0, 0],
                    [ 0, 0, 0, 0, 0, 0],
                    [ 0, 0, 0, 0, 0, 0],
                    [ 1, 2, 3, 4, 5, 6]
                ],
            ]),
            puan.ndarray.integer_ndarray([
                [ 1, 2, 4, 8, 16, 32],
            ]),
            0
        ),
        (
            puan.ndarray.integer_ndarray([
                [ 0, 0, 1, 0, 2, 3],
                [-1,-1,-1,-1, 0, 0],
                [ 0, 0, 1, 2, 0, 0],
                [ 0, 0, 1, 0, 0, 0]
            ]),
            puan.ndarray.integer_ndarray([
                -4, -4, 24, 12, 1, 2
            ]),
            0
        ),
        (
            puan.ndarray.integer_ndarray([
                [
                    [ 0, 0, 1, 0, 2, 3],
                    [-1,-1,-1,-1, 0, 0],
                    [ 0, 0, 1, 2, 0, 0],
                    [ 0, 0, 1, 0, 0, 0]
                ],
                [
                    [ 0, 0, 0, 0, 0, 0],
                    [ 0, 0, 0, 0, 0, 0],
                    [ 0, 0, 0, 0, 0, 0],
                    [ 0, 0, 0, 0, 0, 0],
                ],
                [
                    [ 0, 0, 0, 0, 0, 0],
                    [ 0, 0, 0, 0, 0, 0],
                    [ 0, 0, 0, 0, 0, 0],
                    [ 1, 2, 3, 4, 5, 6]
                ],
                [
                    [-1,-1,-1,-1, 0, 0],
                    [ 0, 0, 1, 0, 2, 3],
                    [ 0, 0, 0, 2, 1, 0],
                    [ 0, 0, 2, 1, 0, 0]
                ],
                [
                    [ 0, 0, 0, 0, 0, 0],
                    [ 1, 4, 7, 0, 0, 0],
                    [ 6, 3,-1, 0, 0, 0],
                    [-3, 0, 7, 0, 0, 0]
                ]
            ]),
            puan.ndarray.integer_ndarray([
                [-4,-4,24,12, 1, 2],
                [ 0, 0, 0, 0, 0, 0],
                [ 1, 2, 4, 8,16,32],
                [-1,-1,24,12, 6, 3],
                [-2, 1, 4, 0, 0, 0]
            ]),
            0
        ),
    ]
    for inpt, expected_output, axis in test_cases:
        actual_output = inpt.ndint_compress(method="shadow", axis=axis)
        assert numpy.array_equal(actual_output, expected_output)

def test_implication_propositions_should_not_share_id():

    a_req_b = pg.Imply("a","b", id="a_req_b").to_dict()
    b_req_c = pg.Imply("b","c", id="b_req_c").to_dict()

    for k,v in a_req_b.items():
        assert (not k in b_req_c) or b_req_c[k] == v, f"{k} is ambivalent: means both {v} and {b_req_c[k]}"

def test_bind_relations_to_compound_id():

    model = pg.All(
        pg.Any("a","b",id="B"),
        pg.Any("c","d",id="C"),
        pg.Any("C","B",id="A"),
        id="model"
    )
    assert len(model.propositions) == 3

def test_dont_override_propositions():

    model = pg.All(
        pg.Imply(pg.Proposition(id="xor_xy", dtype=0, virtual=True), pg.All(*"abc")),
        pg.Xor("x","y", id="xor_xy"),
    )
    assert model.xor_xy.virtual == next(filter(lambda x: x.id == "xor_xy", model.to_polyhedron().variables)).virtual, f"models 'xor_xy' is virtual while models polyhedrons 'xor_xy' is not"

def test_reduce_columns_with_column_variables():

    ph = puan.ndarray.ge_polyhedron([
            [ 1, 1, 1, 0, 0],
            [ 0, 0, 1,-2, 1],
            [-1, 0,-1,-1,-1],
        ],
        puan.variable.from_strings(*"0abcd")
    )
    ph_red = ph.reduce_columns(ph.A.construct(*{"a": 1}.items(), default_value=numpy.nan, dtype=float))
    assert not any((v.id == "a" for v in ph_red.index))
    assert ph_red.shape[0] == 3
    assert ph_red.shape[1] == 4
    assert ph_red.A.shape[1] == ph_red.A.variables.size

def test_reduce_rows_with_variable_index():

    ph = puan.ndarray.ge_polyhedron([
            [ 1, 1, 1, 0, 0],
            [ 0, 0, 1,-2, 1],
            [-1, 0,-1,-1,-1],
        ],
        puan.variable.from_strings(*"0abcd"),
        puan.variable.from_strings(*"ABC"),
    )
    ph_red = ph.reduce_rows([1,0,0])
    assert not any((v.id == "A" for v in ph_red.index))
    assert ph_red.shape[0] == 2
    assert ph_red.shape[1] == 5
    assert ph_red.shape[0] == ph_red.index.size

def test_assume_when_xor_on_top_propositions():

    model = pg.All(
        pg.Xor("p","q", id="B"),
        pg.Imply(
            pg.All("p","g", id="E"),
            pg.All("x","y","z", id="F"),
            id="C",
        ),
        pg.Imply(
            pg.All("q","g", id="G"),
            pg.All("a","b","c", id="H"),
            id="D",
        ),
        id="A",
    )

    expected_assumed_model = pg.All(
        pg.Imply(
            pg.All("g", id="E"),
            pg.All("x","y","z", id="F"),
        ),
        pg.Any(
            "G",
            pg.All("a","b","c", id="H"),
            id="D"
        ),
        id="A",
    )

    # when assuming p here we see that q cannot ever be selected (since first rule will be broken). Therefore, third proposition
    # will always be true (q=0 implies All("q","h") = False and therefore pg.Imply(pg.All("q","h"), pg.All("a","y","c"))) = True)
    # I.e pg.Imply("g", pg.All("x","y","z")) should be left in model and h,a,b,c will be unbound
    actual_assumed_model, _ = model.assume({"p": 1})
    assert actual_assumed_model == expected_assumed_model

def test_logicfunc():
    test_cases = [
        (
            # Relation any between two constraints of types 1 1 (Any Any)
            ((1, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0), (5, 6), (1, 1), 1, 7)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0), (1,2,3,5,6), (1,1,1,1,1), 1, 0)], 0)
        ),
        (
            # Relation any between two constraints of types 1 2 (Any Atmost N-1)
            ((1, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -3, 9)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0, 0, 0), (1,2,3,5,6,7,8), (4,4,4,-1,-1,-1,-1), 0, 0)], 0)
        ),
        (
            # Relation any between two constraints of types 1 3 (Any None)
            ((1, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), 0, 9)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0, 0 ,0), (1,2,3,5,6,7,8), (4,4,4,-1,-1,-1,-1), 0, 0)], 0)
        ),
        (
            # Relation any between two constraints of types 1 4 (Any All)
            ((1, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 4, 9)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0, 0, 0), (1,2,3,5,6,7,8), (4,4,4,1,1,1,1), 4, 0)], 0)
        ),
        (
            # Relation any between two constraints of types 1 5 (Any Atleast n)
            ((1, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 2, 9)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0, 0, 0), (1,2,3,5,6,7,8), (4,4,4,1,1,1,1), 2, 0)], 0)
        ),
        (
            # Relation any between two constraints of types 1 6 (Any Atmost n)
            ((1, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 9)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0, 0, 0), (1,2,3,5,6,7,8), (4,4,4,-1,-1,-1,-1), -2, 0)], 0)
        ),
        (
            # Relation any between two constraints of types 2 2 (Atmost N-1 Atmost N-1)
            ((1, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), -2, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -3, 9)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0, 0, 0), (1,2,3,5,6,7,8), (-4,-4,-4,-1,-1,-1,-1), -15, 0)], 0)
        ),
        (
            # Relation any between two constraints of types 2 3 (Atmost N-1 None)
            ((1, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), -2, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), 0, 9)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0, 0, 0), (1,2,3,5,6,7,8), (-1,-1,-1,-3,-3,-3,-3), -12, 0)], 0)
        ),
        (
            # Relation any between two constraints of types 2 4 (Atmost N-1 All)
            ((1, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), -2, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 4, 9)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0, 0, 0), (1,2,3,5,6,7,8), (-4,-4,-4,1,1,1,1), -8, 0)], 0)
        ),
        (
            # Relation any between two constraints of types 2 5 (Atmost N-1 Atleast n)
            ((1, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), -2, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 2, 9)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0, 0, 0), (1,2,3,5,6,7,8), (-4,-4,-4,1,1,1,1), -10, 0)], 0)
        ),
        (
            # Relation any between two constraints of types 2 6 (Atmost N-1 Atmost n)
            ((1, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), -2, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 9)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0, 0, 0), (1,2,3,5,6,7,8), (-4,-4,-4,-1,-1,-1,-1), -14, 0)], 0)
        ),
        (
            # Relation any between two constraints of types 3 3 (None None)
            ((1, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), 0, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), 0, 9)], 0), True),
            # Not possible to merge to one constraint, but could be merged to min(n1,n2) constraints without introducing new variable
            (3, 1, [
                ((0, 0, 0, 0, 0), (1,5,6,7,8), (-4,-1,-1,-1,-1), -4, 0),
                ((0, 0, 0, 0, 0), (2,5,6,7,8), (-4,-1,-1,-1,-1), -4, 0),
                ((0, 0, 0, 0, 0), (3,5,6,7,8), (-4,-1,-1,-1,-1), -4, 0)], 0)
        ),
        (
            # Relation any between two constraints of types 3 4 (None All)
            ((1, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), 0, 4), ((0, 0), (5, 6), (1, 1), 2, 9)], 0), True),
            # Not possible to merge to one constraint, but could be merged to min(n1,n2) constraints without introducing new variable
            (2, 1, [
                ((0, 0, 0, 0), (5,1,2,3), (3,-1,-1,-1), 0, 0),
                ((0, 0, 0, 0), (6,1,2,3), (3,-1,-1,-1), 0, 0)], 0)
        ),
        (
            # Relation any between two constraints of types 3 4 (None All)
            ((1, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), 0, 4), ((0, 0), (5, 6), (1, 1), 2, 9)], 0), False),
            # Not possible to merge to one constraint
            (3, 1, [
                ((0, 0), (4,9), (1,1), 1, 0),
                ((0, 0, 0, 0), (1,2,3,4), (-1,-1,-1,-3), -3, 4),
                ((0, 0, 0), (5,6,9), (1,1,-2), 0, 9)], 0)
        ),
        (
            # Relation any between two constraints of types 3 5 (None Atleast n)
            ((1, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), 0, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 2, 9)], 0), True),
            # Not possible to merge to one constraint
            (3, 1, [
                ((0, 0, 0, 0, 0), (1,5,6,7,8), (-4,1,1,1,1), -2, 0),
                ((0, 0, 0, 0, 0), (2,5,6,7,8), (-4,1,1,1,1), -2, 0),
                ((0, 0, 0, 0, 0), (3,5,6,7,8), (-4,1,1,1,1), -2, 0),], 0)
        ),
        (
            # Relation any between two constraints of types 3 6 (None Atmost n)
            ((1, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), 0, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 9)], 0), True),
            # Not possible to merge to one constraint
            (3, 1, [
                ((0, 0, 0, 0, 0), (1,5,6,7,8), (-4,-1,-1,-1,-1), -6, 0),
                ((0, 0, 0, 0, 0), (2,5,6,7,8), (-4,-1,-1,-1,-1), -6, 0),
                ((0, 0, 0, 0, 0), (3,5,6,7,8), (-4,-1,-1,-1,-1), -6, 0),], 0)
        ),
        (
            # Relation any between two constraints of types 4 4 (All All)
            ((1, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 3, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 4, 9)], 0), True),
            # Not possible to merge to one constraint
            (3, 1, [
                ((0, 0, 0, 0, 0), (1,5,6,7,8), (4,1,1,1,1), 4, 0),
                ((0, 0, 0, 0, 0), (2,5,6,7,8), (4,1,1,1,1), 4, 0),
                ((0, 0, 0, 0, 0), (3,5,6,7,8), (4,1,1,1,1), 4, 0),], 0)
        ),
        (
            # Relation any between two constraints of types 4 5 (All Atleast n)
            ((1, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 3, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 2, 9)], 0), True),
            # Not possible to merge to one constraint
            (3, 1, [
                ((0, 0, 0, 0, 0), (1,5,6,7,8), (4,1,1,1,1), 2, 0),
                ((0, 0, 0, 0, 0), (2,5,6,7,8), (4,1,1,1,1), 2, 0),
                ((0, 0, 0, 0, 0), (3,5,6,7,8), (4,1,1,1,1), 2, 0),], 0)
        ),
        (
            # Relation any between two constraints of types 4 6 (All Atmost n)
            ((1, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 3, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 9)], 0), True),
            # Not possible to merge to one constraint
            (3, 1, [
                ((0, 0, 0, 0, 0), (1,5,6,7,8), (4,-1,-1,-1,-1), -2, 0),
                ((0, 0, 0, 0, 0), (2,5,6,7,8), (4,-1,-1,-1,-1), -2, 0),
                ((0, 0, 0, 0, 0), (3,5,6,7,8), (4,-1,-1,-1,-1), -2, 0),], 0)
        ),
        (
            # Relation any between two constraints of types 5 5 (Atleast n Atleast n)
            ((1, 1, [((0, 0, 0, 0), (1, 2, 3, 4), (1, 1, 1, 1), 2, 9), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 2, 10)], 0), True),
            # Not possible to merge to one constraint
            (3, 1, [
                ((0, 0), (9, 10), (1,1), 1, 0),
                ((0, 0, 0, 0, 0), (1,2,3,4,9), (1,1,1,1,-2), 0, 9),
                ((0, 0, 0, 0, 0), (5,6,7,8,10), (1,1,1,1,-2), 0, 10),], 0)
        ),
        (
            # Relation any between two constraints of types 5 6 (Atleast n Atmost n)
            ((1, 1, [((0, 0, 0, 0), (1, 2, 3, 4), (1, 1, 1, 1), 2, 9), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 10)], 0), True),
            # Not possible to merge to one constraint
            (3, 1, [
                ((0, 0), (9, 10), (1,1), 1, 0),
                ((0, 0, 0, 0, 0), (1,2,3,4,9), (1,1,1,1,-2), 0, 9),
                ((0, 0, 0, 0, 0), (5,6,7,8,10), (-1,-1,-1,-1,-2), -4, 10),], 0)
        ),
        (
            # Relation any between two constraints of types 6 6 (Atmost n Atmost n)
            ((1, 1, [((0, 0, 0, 0), (1, 2, 3, 4), (-1, -1, -1, -1), -2, 9), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 10)], 0), True),
            # Not possible to merge to one constraint
            (3, 1, [
                ((0, 0), (9, 10), (1,1), 1, 0),
                ((0, 0, 0, 0, 0), (1,2,3,4,9), (-1,-1,-1,-1,-2), -4, 9),
                ((0, 0, 0, 0, 0), (5,6,7,8,10), (-1,-1,-1,-1,-2), -4, 10),], 0)
        ),
        (
            # Relation any between four constraints of types 1 (Any)
            ((1, 1, [
                ((0,), (1,), (1,), 1, 5),
                ((0,), (2,), (1,), 1, 6),
                ((0,), (3,), (1,), 1, 7),
                ((0,), (4,), (1,), 1, 8)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0), (1, 2, 3, 4), (1,1,1,1), 1, 0),], 0)
        ),
        (
            # Relation all between two constraints of types 1 1 (Any Any)
            ((2, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0), (5, 6), (1, 1), 1, 7)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [
                ((0, 0, 0, 0), (5,1,2,3), (3,1,1,1), 4, 0),
                ((0, 0, 0, 0), (6,1,2,3), (3,1,1,1), 4, 0)], 0)
        ),
        (
            # Relation all between two constraints of types 1 2 (Any Atmost N-1)
            ((2, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -3, 9)], 0), True),
            # Possible to merge to one constraint
            (3, 1, [
                ((0, 0), (4, 9), (1,1), 2, 0),
                ((0, 0, 0, 0), (1,2,3,4), (1, 1, 1,-1), 0, 4),
                ((0,0,0,0,0),(5,6,7,8,9),(-1, -1, -1, -1, -1), -4, 9)], 0)
        ),
        (
            # Relation all between two constraints of types 1 3 (Any None)
            ((2, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), 0, 9)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0, 0 ,0), (1,2,3,5,6,7,8), (1,1,1,3,3,3,3), 1, 0)], 0)
        ),
        (
            # Relation all between two constraints of types 1 4 (Any All)
            ((2, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 4, 9)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0, 0, 0), (1,2,3,5,6,7,8), (1,1,1,3,3,3,3), 13, 0)], 0)
        ),
        (
            # Relation all between two constraints of types 1 5 (Any Atleast n)
            ((2, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 2, 9)], 0), True),
            # Not possible to merge to one constraint
            (1, 1, [
                ((0, 0, 0, 0, 0), (1,5,6,7,8), (4,1,1,1,1), 6, 0),
                ((0, 0, 0, 0, 0), (2,5,6,7,8), (4,1,1,1,1), 6, 0),
                ((0, 0, 0, 0, 0), (3,5,6,7,8), (4,1,1,1,1), 6, 0)], 0)
        ),
        (
            # Relation all between two constraints of types 1 6 (Any Atmost n)
            ((2, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 9)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [
                ((0, 0, 0, 0, 0), (1,5,6,7,8), (4,-1,-1,-1,-1), 2, 0),
                ((0, 0, 0, 0, 0), (2,5,6,7,8), (4,-1,-1,-1,-1), 2, 0),
                ((0, 0, 0, 0, 0), (3,5,6,7,8), (4,-1,-1,-1,-1), 2, 0)], 0)
        ),
        (
            # Relation all between two constraints of types 2 2 (Atmost N-1 Atmost N-1)
            ((2, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), -2, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -3, 9)], 0), True),
            # Not possible to merge to one constraint
            (3, 1, [
                ((0, 0), (4, 9), (1,1), 2, 0),
                ((0, 0, 0, 0), (1,2,3,4), (-1, -1, -1,-1), -3, 4),
                ((0,0,0,0,0),(5,6,7,8,9),(-1, -1, -1, -1, -1), -4, 9)], 0)
        ),
        (
            # Relation all between two constraints of types 2 3 (Atmost N-1 None)
            ((2, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), -2, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), 0, 9)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0, 0, 0), (1,2,3,5,6,7,8), (-1,-1,-1,-3,-3,-3,-3), -2, 0)], 0)
        ),
        (
            # Relation all between two constraints of types 2 4 (Atmost N-1 All)
            ((2, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), -2, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 4, 9)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0, 0, 0), (1,2,3,5,6,7,8), (-1,-1,-1,3,3,3,3), 10, 0)], 0)
        ),
        (
            # Relation all between two constraints of types 2 5 (Atmost N-1 Atleast n)
            ((2, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), -2, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 2, 9)], 0), True),
            # Possible to merge to one constraint
            (3, 1, [
                ((0, 0), (4, 9), (1,1), 2, 0),
                ((0, 0, 0, 0), (1,2,3,4), (-1, -1, -1,-1), -3, 4),
                ((0,0,0,0,0),(5,6,7,8,9),(1, 1, 1, 1, -2), 0, 9)], 0)
        ),
        (
            # Relation all between two constraints of types 2 6 (Atmost N-1 Atmost n)
            ((2, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), -2, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 9)], 0), True),
            # Not possible to merge to one constraint
            (3, 1, [
                ((0, 0), (4, 9), (1,1), 2, 0),
                ((0, 0, 0, 0), (1,2,3,4), (-1, -1, -1,-1), -3, 4),
                ((0,0,0,0,0),(5,6,7,8,9),(-1, -1, -1, -1, -2), -4, 9)
            ], 0)
        ),
        (
            # Relation all between two constraints of types 3 3 (None None)
            ((2, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), 0, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), 0, 9)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0, 0, 0), (1,2,3,5,6,7,8), (-1,-1,-1,-1,-1,-1,-1), 0, 0)], 0)
        ),
        (
            # Relation all between two constraints of types 3 4 (None All)
            ((2, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), 0, 4), ((0, 0), (5, 6), (1, 1), 2, 9)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0), (1,2,3,5,6), (-1,-1,-1,1, 1), 2, 0)], 0)
        ),
        (
            # Relation all between two constraints of types 3 5 (None Atleast n)
            ((2, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), 0, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 2, 9)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0, 0, 0), (1,2,3,5,6,7,8), (-4,-4,-4,1,1,1,1), 2, 0)], 0)
        ),
        (
            # Relation all between two constraints of types 3 6 (None Atmost n)
            ((2, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), 0, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 9)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0, 0, 0), (1,2,3,5,6,7,8), (-4,-4,-4,-1,-1,-1,-1), -2, 0)], 0)
        ),
        (
            # Relation all between two constraints of types 4 4 (All All)
            ((2, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 3, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 4, 9)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0, 0, 0), (1,2,3,5,6,7,8), (1,1,1,1,1,1,1), 7, 0)], 0)
        ),
        (
            # Relation all between two constraints of types 4 5 (All Atleast n)
            ((2, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 3, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 2, 9)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0, 0, 0), (1,2,3,5,6,7,8), (4,4,4,1,1,1,1), 14, 0)], 0)
        ),
        (
            # Relation all between two constraints of types 4 6 (All Atmost n)
            ((2, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 3, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 9)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0, 0, 0), (1,2,3,5,6,7,8), (4,4,4,-1,-1,-1,-1), 10, 0)], 0)
        ),
        (
            # Relation all between two constraints of types 5 5 (Atleast n Atleast n)
            ((2, 1, [((0, 0, 0, 0), (1, 2, 3, 4), (1, 1, 1, 1), 2, 9), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 2, 10)], 0), True),
            # Not possible to merge to one constraint
            (3, 1, [
                ((0, 0), (9, 10), (1,1), 2, 0),
                ((0, 0, 0, 0, 0), (1,2,3,4,9), (1,1,1,1,-2), 0, 9),
                ((0, 0, 0, 0, 0), (5,6,7,8,10), (1,1,1,1,-2), 0, 10),], 0)
        ),
        (
            # Relation all between two constraints of types 5 6 (Atleast n Atmost n)
            ((2, 1, [((0, 0, 0, 0), (1, 2, 3, 4), (1, 1, 1, 1), 2, 9), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 10)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0, 0, 0, 0,0), (1,2,3,4,5,6,7,8), (4,4,4,4,-1,-1,-1,-1), 6, 0)], 0)
        ),
        (
            # Relation all between two constraints of types 6 6 (Atmost n Atmost n)
            ((2, 1, [((0, 0, 0, 0), (1, 2, 3, 4), (-1, -1, -1, -1), -2, 9), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 10)], 0), True),
            # Not possible to merge to one constraint
            (3, 1, [
                ((0, 0), (9, 10), (1,1), 2, 0),
                ((0, 0, 0, 0, 0), (1,2,3,4,9), (-1,-1,-1,-1,-2), -4, 9),
                ((0, 0, 0, 0, 0), (5,6,7,8,10), (-1,-1,-1,-1,-2), -4, 10),], 0)
        ),
        (
            # Relation any between four constraints of types 1 (Any)
            ((4, 1, [
                ((0,), (1,), (1,), 1, 5),
                ((0,), (2,), (1,), 1, 6),
                ((0,), (3,), (1,), 1, 7),
                ((0,), (4,), (1,), 1, 8)], 0), True),
            # Possible to merge to one constraint
            (1, 1, [((0, 0, 0, 0), (4,3,1, 2), (1,1,1,1), 4, 0),], 0)
        )
    ]
    for i in range(10000):
        for inpt, expected_output in test_cases:
            actual_output = lf.transform(*inpt)
            assert expected_output == actual_output

def test_json_conversion():

    model = pg.All(
        pg.Any("x","y", id="B"),
        pg.Any("a","b", id="C"),
        id="A",
    )

    converted = pg.from_json(model.to_json())
    assert model == converted

    model = cc.StingyConfigurator(
        cc.Any("x","y", id="B"),
        cc.Any("a","b", id="C"),
        id="A",
    )

    converted = cc.StingyConfigurator.from_json(model.to_json())
    assert model == converted

def test_json_conversions_with_assume():

    data = {
        "type": "StingyConfigurator",
        "propositions": [
            {
                "type": "Imply",
                "condition": {
                    "type": "Any",
                    "propositions": [
                        {"id": "a"},
                        {"id": "b"}
                    ]
                },
                "consequence": {
                    "type": "Xor",
                    "propositions": [
                        {"id": "x"},
                        {"id": "y"},
                        {"id": "z"}
                    ],
                    "default": ["z"]
                }
            }
        ]
    }
    actual, consequence = cc.StingyConfigurator.from_json(data).assume({"a": 1})
    expected = {
        "type": "StingyConfigurator",
        "propositions": [
            {
                "id": "VAR56ed",
                "type": "AtMost",
                "propositions": [
                    {
                        "id": "x"
                    },
                    {
                        "id": "y"
                    },
                    {
                        "id": "z"
                    }
                ],
                "value": 1
            },
            {
                "id": "VAR5823",
                "type": "Any",
                "propositions": [
                    {
                        "id": "x"
                    },
                    {
                        "id": "y"
                    },
                    {
                        "id": "z"
                    }
                ],
                "default": [
                    "z"
                ]
            }
        ]
    }
    assert consequence['a'] == 1
    assert actual.to_json() == expected

def test_assume_should_keep_full_tree():

    assumed, consequence = cc.StingyConfigurator(
        pg.Imply(
            pg.Any("a","b"),
            cc.Xor(*("xyz"), default="z")
        )
    ).assume({"a": 1})
    prio_candidates = list(filter(lambda x: hasattr(x, "prio"), assumed.propositions))

    assert consequence['a'] == 1
    assert len(prio_candidates) == 1
    assert prio_candidates[0].prio == -1
    assert prio_candidates[0].propositions[1].id == "z"

def test_xnor_proposition():
    
    actual_model = pg.XNor(
        pg.All(
            "x",
            pg.Xor("y","z")
        )
    )

    expected_dict = {
        'VARe546': (1, ['VAR3d36', 'VARc034'], 2, 0, 1),
        'VAR3d36': (1, ['VAR28eb'], 1, 0, 1),
        'VAR28eb': (1, ['VAR41ae', 'x'], 2, 0, 1),
        'VAR41ae': (1, ['VAR077e', 'VARbed7'], 2, 0, 1),
        'VAR077e': (1, ['y', 'z'], 1, 0, 1),
        'y': (1, [], 1, 0, 0),
        'z': (1, [], 1, 0, 0),
        'VARbed7': (-1, ['y', 'z'], -1, 0, 1),
        'x': (1, [], 1, 0, 0),
        'VARc034': (-1, ['VAR28eb'], -1, 0, 1)
    }

    assert actual_model.to_dict() == expected_dict

def test_proposition_polyhedron_conversions():

    actual = pg.All(
        pg.Not("x"),
        id="all_not"
    ).to_polyhedron(True)
    assert not all(actual.A.dot(actual.construct(*{"x": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.construct(*{"VARc96e": 1}.items())) >= actual.b)

    actual = pg.Not(
        pg.All("x","y","z", id="all_xyz")
    ).to_polyhedron(True)
    assert all(actual.A.dot(actual.construct(*{"x": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.construct(*{"x": 1, "y": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.construct(*{"x": 1, "y": 1, "z": 1}.items())) >= actual.b)

    actual = pg.Not(
        pg.Any("x","y","z")
    ).to_polyhedron(True)
    assert not all(actual.A.dot(actual.construct(*{"x": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.construct(*{"y": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.construct(*{"z": 1}.items())) >= actual.b)

    actual = pg.Imply(
        condition=pg.Not("x"),
        consequence=pg.All("a","b","c")
    ).to_polyhedron(True)
    assert all(actual.A.dot(actual.construct(*{"VARfe37": 1, "x": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.construct(*{"VARa786": 1, "a": 1, "b": 1, "c": 1}.items())) >= actual.b)

    actual = pg.Not(
        pg.Imply(
            condition=pg.All("x","y","z"), 
            consequence=pg.Any("a","b","c")
        ),
    ).to_polyhedron(True)
    assert all(actual.A.dot(actual.construct(*{"VARb6a0": 1, "VARe391": 1, "x": 1, "y": 1, "z": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.construct(*{"VARb6a0": 1, "VARe391": 1, "x": 1, "y": 1, "z": 1, "a": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.construct(*{"VARb6a0": 1, "VARe391": 1, "x": 1, "y": 1, "z": 1, "b": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.construct(*{"VARb6a0": 1, "VARe391": 1, "x": 1, "y": 1, "z": 1, "c": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.construct(*{"VARb6a0": 1, "VARe391": 1, "x": 1, "y": 1}.items())) >= actual.b)

    actual = pg.Not(
        pg.AtLeast("x","y","z", value=2)
    ).to_polyhedron(True)
    assert not all(actual.A.dot(actual.construct(*{"x": 1, "y": 1, "z": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.construct(*{"x": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.construct(*{"y": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.construct(*{"z": 1}.items())) >= actual.b)

    actual = pg.Not(
        pg.AtMost("x","y","z", value=2)
    ).to_polyhedron(True)
    assert not all(actual.A.dot(actual.construct(*{"x": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.construct(*{"y": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.construct(*{"z": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.construct(*{"x": 1, "y": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.construct(*{"x": 1, "z": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.construct(*{"y": 1, "z": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.construct(*{"x": 1, "y": 1, "z": 1}.items())) >= actual.b)

def test_assuming_integer_variables():

    """
        Assumes x,y,z and we'll expect that t must be between
        10 and 20.
    """
    model = pg.Imply(
        pg.All(*"xyz"),
        pg.All(
            pg.AtLeast(
                pg.Integer("t"),
                value=10,
                id="at-least-10"
            ),
            pg.AtMost(
                pg.Integer("t"),
                value=20,
                id="at-most-20"
            )
        ), 
    )

    assumed = model.assume({"x": 1, "y": 1, "z": 1})[0].to_polyhedron(True)
    assert all(assumed.A.dot(assumed.construct(*{"at-least-10": 1, "at-most-20": 1, "t": 10}.items())) >= assumed.b)
    assert all(assumed.A.dot(assumed.construct(*{"at-least-10": 1, "at-most-20": 1, "t": 20}.items())) >= assumed.b)
    assert not all(assumed.A.dot(assumed.construct(*{"t": 1}.items())) >= assumed.b)
    assert not all(assumed.A.dot(assumed.construct(*{"t": -22}.items())) >= assumed.b)
    assert not all(assumed.A.dot(assumed.construct(*{"t": 9}.items())) >= assumed.b)
    assert not all(assumed.A.dot(assumed.construct(*{"t": 21}.items())) >= assumed.b)

    model = pg.Imply(
        pg.All(
            pg.AtLeast(
                pg.Integer("t"),
                value=10,
                id="at-least-10"
            ),
            pg.AtMost(
                pg.Integer("t"),
                value=20,
                id="at-most-20"
            )
        ), 
        pg.All(*"xyz", id="all-xyz")
    )

    assumed_model, assumed_items = model.assume({"t": 12})
    assert assumed_model.is_tautologi
    assert assumed_items['t'] == 12
    assert assumed_items['x'] == 1
    assert assumed_items['y'] == 1
    assert assumed_items['z'] == 1

def test_bound_approx():

    # Test random constraints with a being an integer
    variables = [
        puan.variable("0", 1, True),
        puan.variable("a", 1, False),
        puan.variable("b", 0, False),
        puan.variable("c", 0, False),
        puan.variable("d", 0, False),
    ]
    actual = puan.ndarray.ge_polyhedron([
        [ 3, 1, 1, 1, 0],
        [ 3, 1, 0, 0, 0],
        [ 0,-2, 1, 1, 0],
        [-1,-1,-1, 0, 0],
    ], variables=variables).bounds_approx((-10,10))
    expected = numpy.array([
        [ 3, 0, 0, 0],
        [ 1, 1, 1, 1]
    ])
    assert (actual == expected).all()


    # Test integer lower bound will increase
    variables = [
        puan.variable("0", 1, True),
        puan.variable("a", 1, False),
        puan.variable("b", 0, False),
        puan.variable("c", 0, False),
        puan.variable("d", 0, False),
    ]
    actual = puan.ndarray.ge_polyhedron([
        [ 3, 1, 0, 0, 0],
        [ 0,-2, 1, 1, 0],
    ], variables=variables).bounds_approx((-10,10))
    expected = numpy.array([
        [3,0,0,0],
        [1,1,1,1]
    ])
    assert (actual == expected).all()
    
    # Test that ordinary bounds are kept
    actual = puan.ndarray.ge_polyhedron([
        [ 0,-2, 1, 1, 0],
        [-3,-2,-1,-1, 0],
        [-3,-2,-2, 1, 1]
    ]).bounds_approx((-10,10))
    expected = numpy.array([
        [0,0,0,0],
        [1,1,1,1]
    ])
    assert (actual == expected).all()
    
    # Test force lower bound to increase to 1
    actual = puan.ndarray.ge_polyhedron([
        [3,1,1,1,0]
    ]).bounds_approx((-10,10))
    expected = numpy.array([
        [1,1,1,0],
        [1,1,1,1]
    ])
    assert (actual == expected).all()
    
    # Test lower bound won't increase while at least one must be set
    actual = puan.ndarray.ge_polyhedron([
        [1,1,1,1,0]
    ]).bounds_approx((-10,10))
    expected = numpy.array([
        [0,0,0,0],
        [1,1,1,1]
    ])
    assert (actual == expected).all()
    
    # Test integer upper bound will decrease
    variables = [
        puan.variable("0", 1, True),
        puan.variable("a", 1, False),
        puan.variable("b", 0, False),
        puan.variable("c", 0, False),
        puan.variable("d", 0, False),
    ]
    actual = puan.ndarray.ge_polyhedron([
        [-10,-1, 0, 0, 0],
        [  0,-2, 1, 1, 0],
    ], variables=variables).bounds_approx((-10,10))
    expected = numpy.array([
        [-10,0,0,0],
        [ 1,1,1,1]
    ])
    assert (actual == expected).all()    

    # Test "inverted" boolean (as integers)
    variables = [
        puan.variable("0", 1, True),
        puan.variable("a", 1, False),
        puan.variable("b", 1, False),
        puan.variable("c", 1, False),
        puan.variable("d", 1, False),
    ]
    actual = puan.ndarray.ge_polyhedron([
        [ 4,-1,-1,-1,-1],
    ], variables=variables).bounds_approx((-1,0))
    expected = numpy.array([
        [-1,-1,-1,-1],
        [-1,-1,-1,-1],
    ])
    assert (actual == expected).all()    

    # Test if finding lower bound < 0 
    variables = [
        puan.variable("0", 1, True),
        puan.variable("a", 1, False),
        puan.variable("b", 0, False)
    ]
    actual = puan.ndarray.ge_polyhedron([
        [ 1,-1,-1],
    ], variables=variables).bounds_approx((-5,0))
    expected = numpy.array([
        [-5,0],
        [-1,0],
    ])
    assert (actual == expected).all()    

def test_bounds_init(): 

    variables = [
        puan.variable("0", 1, True),
        puan.variable("a", 1, False),
        puan.variable("b", 0, False),
        puan.variable("c", 1, False),
        puan.variable("d", 0, False),
    ]

    actual = puan.ndarray.ge_polyhedron([
        [ 1,-1,-1,-1,-1],
    ], variables=variables).bounds_init((-5,0))
    expected = numpy.array([
        [-5, 0,-5, 0],
        [ 0, 0, 0, 0]
    ])
    assert (actual == expected).all()