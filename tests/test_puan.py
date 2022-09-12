import ast
import functools
import itertools
import puan
import puan.misc
import puan.ndarray
import puan.logic
import puan.logic.logicfunc as lf
import puan.logic.plog as pg
import puan.modules.configurator as cc
import puan.solver
import numpy
import operator
import maz
import math

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
        original_interpretation_vector = original_polyhedron.A.construct(*random_interpretation.items())
        assumed_interpretation_vector = assumed_polyhedron.A.construct(*random_interpretation.items())

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

    expected_feasible_configurations = matrix.A.construct(
        [("B",1),("D",1),("E",1),("F",1)],
        [("B",1),("C",1),("x",1),("y",1)],
        [("B",1),("C",1),("a",1),("b",1),("x",1),("y",1)],
        [("B",1),("D",1),("E",1),("F",1),("a",1),("c",1)]
    )
    expected_infeasible_configurations = matrix.A.construct(
       #"a  b  c  d  x  y"
       [],
       [("B",1),("C",1),("E",1),("F",1),("a",1),("b",1)],
       [("B",1),("C",1),("E",1),("F",1),("c",1),("d",1)]
    )

    eval_fn = maz.compose(all, functools.partial(operator.le, matrix.b), matrix.A.dot)
    assert all(map(eval_fn, expected_feasible_configurations))
    assert not any(map(eval_fn, expected_infeasible_configurations))

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
            ((2, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 3, 4), ((0, 0, 0), (1, 5, 6), (1, 1, 1), 3, 7)], 0), True),
            (1, 1, [((0,0,0,0,0),(1,2,3,5,6), (2,1,1,1,1), 6, 0)], 0)
        ),
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
            # Not possible to merge to one constraint
            (1, 1, [
                ((0, 0, 0, 0), (5,1,2,3), (3,1,1,1), 4, 0),
                ((0, 0, 0, 0), (6,1,2,3), (3,1,1,1), 4, 0)], 0)
        ),
        (
            # Relation all between two constraints of types 1 2 (Any Atmost N-1)
            ((2, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -3, 9)], 0), True),
            # Not ossible to merge to one constraint
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
            # Not possible to merge to one constraint
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
            # Not possible to merge to one constraint
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
            (1, 1, [((0, 0, 0, 0, 0, 0, 0,0), (1,2,3,4,5,6,7,8), (4,4,4,4,-1,-1,-1,-1), 6, 0)], 1)
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
    # for i in range(10000):
    for inpt, expected_output in test_cases[:1]:
        print(inpt)
        actual_output = lf.transform(*inpt)
        assert expected_output == actual_output

def test_json_conversion():

    model = pg.All(
        pg.Any("x","y"),
        pg.Any("a","b")
    )

    converted = pg.from_json(model.to_json())
    assert model == converted

    model = cc.StingyConfigurator(
        cc.Any("x","y"),
        cc.Any("a","b")
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
    _, actual = cc.StingyConfigurator.from_json(data).assume({"a": 1})
    expected = {
        'a': 1,
        'VAR065d624e3550b8dff7e18ad457d19a628fb05e77b6c8762f04a7858ec5b77484':1,
        'VAR1ac38f14258627b4813573deb4795cfc22545c07f69d00fd713da9755ce76811':0,
        'VAR56ed3c3d0f7773268a7aaebc95964ef976f28809a4e9a7b4f0e5a831ae3ac9c7':1,
        'VAR9a9c96dd78e211a880ee2dbf7728d697001b54c7db0a6d20a460c1ea82bae29e':1,
        'VARc4a183aab2c8ed78438a42ff8a3321f15c2f312d2a3a347b4906e302f64bc769':1,
    }
    assert actual == expected

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
        "VARa060e265bd4909b52363fbdb8d8568848c901d6997442dcb4f9ead8f8f60cd5c": (
            1,
            [
                "VAR1e4bb26b642521f5a43305536694f4e93ca99fa29761f29773c80e70c40fc72a",
                "VAR2ea7bec55a9805ff47ffd3dda7652d801851f657f1052fac6b3322e90193597e"
            ],
            2,
            0,
            1
        ),
        "VAR1e4bb26b642521f5a43305536694f4e93ca99fa29761f29773c80e70c40fc72a": (
            -1,
            [
                "VAR0ef6d9242bb5e856fbfd6aef974e00cea8ff805b4cab302a20afaa5452354d17"
            ],
            -1,
            0,
            1
        ),
        "VAR0ef6d9242bb5e856fbfd6aef974e00cea8ff805b4cab302a20afaa5452354d17": (
            1,
            [
                "VARd61375cf2826250cd27eb8fa6025c5ac42a64ea84ac5eacbc07703f7b19c9c7a",
                "x"
            ],
            2,
            0,
            1
        ),
        "VARd61375cf2826250cd27eb8fa6025c5ac42a64ea84ac5eacbc07703f7b19c9c7a": (
            1,
            [
                "VAR077e3a5a7175fc147a042f0d32e6ac4a5119fc14731e8602fb077a155f77b68b",
                "VARbed7381cd125603a77b3534b72c910e4ba98d0c9ad052cdd27ab9518eac055ea"
            ],
            2,
            0,
            1
        ),
        "VAR077e3a5a7175fc147a042f0d32e6ac4a5119fc14731e8602fb077a155f77b68b": (
            1,
            [
                "y",
                "z"
            ],
            1,
            0,
            1
        ),
        "y": (1,[],1,0,0),
        "z": (1,[],1,0,0),
        "VARbed7381cd125603a77b3534b72c910e4ba98d0c9ad052cdd27ab9518eac055ea": (
            -1,
            ["y","z"],
            -1,
            0,
            1
        ),
        "x": (1,[],1,0,0),
        "VAR2ea7bec55a9805ff47ffd3dda7652d801851f657f1052fac6b3322e90193597e": (
            1,
            [
                "VAR0ef6d9242bb5e856fbfd6aef974e00cea8ff805b4cab302a20afaa5452354d17"
            ],
            1,
            0,
            1
        )
    }

    assert actual_model.to_dict() == expected_dict

def test_proposition_polyhedron_conversions():

    actual = pg.All(
        pg.Not("x"),
        id="all_not"
    ).to_polyhedron(True)
    assert not all(actual.A.dot(actual.A.construct(*{"x": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.A.construct(*{"VARc96efc8ea4acc75f6dbddd0acac8f189b4c566f77b76b6299161a14e4eeb2caf": 1}.items())) >= actual.b)

    actual = pg.Not(
        pg.All("x","y","z", id="all_xyz")
    ).to_polyhedron(True)
    assert all(actual.A.dot(actual.A.construct(*{"x": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.A.construct(*{"x": 1, "y": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"x": 1, "y": 1, "z": 1}.items())) >= actual.b)

    actual = pg.Not(
        pg.Any("x","y","z")
    ).to_polyhedron(True)
    assert not all(actual.A.dot(actual.A.construct(*{"x": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"y": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"z": 1}.items())) >= actual.b)

    actual = pg.Imply(
        condition=pg.Not("x"),
        consequence=pg.All("a","b","c")
    ).to_polyhedron(True)
    assert all(actual.A.dot(actual.A.construct(*{"VARfe372293ac6fc8767d248278e9ceacbb53aa57de8d3b30ef20813933935d1332": 1, "x": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.A.construct(*{"VARa786dc00ea76fe754d21e63ec49ef338cc4771c44ed9562a0fb0bd52d305ae1e": 1, "a": 1, "b": 1, "c": 1}.items())) >= actual.b)

    actual = pg.Not(
        pg.Imply(
            condition=pg.All("x","y","z"), 
            consequence=pg.Any("a","b","c")
        ),
    ).to_polyhedron(True)
    assert all(actual.A.dot(actual.A.construct(*{"VARb6a05d7d91efc84e49117524cffa01cba8dcb1f14479be025342b909c9ab0cc2": 1, "VARe3918cdbd4ac804be32d2b5a3f2890d6ae5f6d3fb9246b429be1bb973edd157a": 1, "x": 1, "y": 1, "z": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"VARb6a05d7d91efc84e49117524cffa01cba8dcb1f14479be025342b909c9ab0cc2": 1, "VARe3918cdbd4ac804be32d2b5a3f2890d6ae5f6d3fb9246b429be1bb973edd157a": 1, "x": 1, "y": 1, "z": 1, "a": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"VARb6a05d7d91efc84e49117524cffa01cba8dcb1f14479be025342b909c9ab0cc2": 1, "VARe3918cdbd4ac804be32d2b5a3f2890d6ae5f6d3fb9246b429be1bb973edd157a": 1, "x": 1, "y": 1, "z": 1, "b": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"VARb6a05d7d91efc84e49117524cffa01cba8dcb1f14479be025342b909c9ab0cc2": 1, "VARe3918cdbd4ac804be32d2b5a3f2890d6ae5f6d3fb9246b429be1bb973edd157a": 1, "x": 1, "y": 1, "z": 1, "c": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"VARb6a05d7d91efc84e49117524cffa01cba8dcb1f14479be025342b909c9ab0cc2": 1, "VARe3918cdbd4ac804be32d2b5a3f2890d6ae5f6d3fb9246b429be1bb973edd157a": 1, "x": 1, "y": 1}.items())) >= actual.b)

    actual = pg.Not(
        pg.AtLeast("x","y","z", value=2)
    ).to_polyhedron(True)
    assert not all(actual.A.dot(actual.A.construct(*{"x": 1, "y": 1, "z": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.A.construct(*{"x": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.A.construct(*{"y": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.A.construct(*{"z": 1}.items())) >= actual.b)

    actual = pg.Not(
        pg.AtMost("x","y","z", value=2)
    ).to_polyhedron(True)
    assert not all(actual.A.dot(actual.A.construct(*{"x": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"y": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"z": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"x": 1, "y": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"x": 1, "z": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"y": 1, "z": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.A.construct(*{"x": 1, "y": 1, "z": 1}.items())) >= actual.b)

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
    assert all(assumed.A.dot(assumed.A.construct(*{"at-least-10": 1, "at-most-20": 1, "t": 10}.items())) >= assumed.b)
    assert all(assumed.A.dot(assumed.A.construct(*{"at-least-10": 1, "at-most-20": 1, "t": 20}.items())) >= assumed.b)
    assert not all(assumed.A.dot(assumed.A.construct(*{"t": 1}.items())) >= assumed.b)
    assert not all(assumed.A.dot(assumed.A.construct(*{"t": -22}.items())) >= assumed.b)
    assert not all(assumed.A.dot(assumed.A.construct(*{"t": 9}.items())) >= assumed.b)
    assert not all(assumed.A.dot(assumed.A.construct(*{"t": 21}.items())) >= assumed.b)

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

def test_plog_not_json_should_only_accept_single_proposition():

    try:
        # Should not be allowed
        puan.logic.plog.Not("x", "y").to_json()
        assert False
    except:
        pass

    try:
        # Should not be allowed
        puan.logic.plog.from_json({
            "type": "Not",
            "propositions": [
                {"id": "x"},
                {"id": "y"},
                {"id": "z"},
            ]
        })
        assert False
    except:
        pass

    expected_model = puan.logic.plog.Not(
        puan.logic.plog.All(*"xyz")
    )
    actual_model = puan.logic.plog.from_json({
        "type": "Not",
        "proposition": {
            "type": "All",
            "propositions": [
                {"id": "x"},
                {"id": "y"},
                {"id": "z"},
            ]
        }
    })

    assert expected_model == actual_model

def test_constructing_empty_array():

    polyhedron = puan.ndarray.ge_polyhedron([[0,-2,1,1,0]], puan.variable.from_strings(*"0abcd"))
    arr = polyhedron.construct(*{}.items(), default_value=numpy.nan, dtype=float)
    assert numpy.isnan(arr).all()

def test_id_should_not_be_returned_when_proposition_is_virtual():

    model = pg.Imply(
        pg.All("a"),
        pg.Xor(*"xyz")
    )
    assumed = model.assume({"a": 1})[0]
    d = assumed.to_json()
    assert all(
        map(
            maz.compose(
                operator.not_,
                maz.pospartial(operator.contains, [(1, "id")])
            ), 
            d['propositions']
        )
    )

def test_configuring_using_ge_polyhedron_config():

    model = cc.StingyConfigurator(
        pg.Imply(
            pg.All(*"ab"),
            cc.Xor(*"xyz", default="z")
        ),
        pg.Any(*"pqr")
    )

    def dummy_solver(A, b, ints, objs):
        return numpy.ones((objs.shape[0], A.shape[1]))

    expected = [
        puan.SolutionVariable("a",0,False,1.0),
        puan.SolutionVariable("b",0,False,1.0),
        puan.SolutionVariable("p",0,False,1.0),
        puan.SolutionVariable("q",0,False,1.0),
        puan.SolutionVariable("r",0,False,1.0),
        puan.SolutionVariable("x",0,False,1.0),
        puan.SolutionVariable("y",0,False,1.0),
        puan.SolutionVariable("z",0,False,1.0),
    ]
    actual = list(model.select({"a": 1}, solver=dummy_solver, include_virtual_vars=False))
    assert actual[0] == expected

def test_dump_load_ge_polyhedron_config():

    model = cc.StingyConfigurator(
        pg.Imply(
            pg.All(*"ab"),
            cc.Xor(*"xyz", default="z")
        ),
        pg.Any(*"pqr")
    )

    def dummy_solver(A, b, ints, objs):
        return numpy.ones((objs.shape[0], A.shape[1]))

    expected = puan.ndarray.ge_polyhedron_config.from_b64(
        model.polyhedron.to_b64()
    ).select({"a": 1}, solver=dummy_solver)
    actual = model.select({"a": 1}, solver=dummy_solver)

    assert list(actual) == list(expected)

def test_default_prio_vector_weights():

    """
        Test is here to see that there's exactly one prio with -2,
        which represent choosing A and x together
    """
    
    model = cc.StingyConfigurator(
        pg.Imply(
            pg.All("A"),
            cc.Xor(*"xy", default="y")
        ),
        pg.Imply(
            pg.All("A"),
            cc.Xor(*"xz", default="z")
        ),
    )
    
    assert len(list(filter(functools.partial(operator.eq, -2), model.default_prios.values()))) == 1

def test_logicfunc_with_truth_table():
    constraints = [
            ((2, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 3, 6), ((0, 0, 0), (1, 0, 4), (1, 1, 1), 3, 7)], 0), True),
            ((1, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0), (1, 6), (1, 1), 1, 7)], 0), True),
            ((1, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), -2, 4), ((0, 0, 0, 0), (1, 2, 3, 8), (-1, -1, -1, -1), 0, 9)], 0), True),
            ((1, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0), (5, 6), (1, 1), 1, 7)], 0), True),
            ((1, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -3, 9)], 0), True),
            ((1, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), 0, 9)], 0), True),
            ((1, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 4, 9)], 0), True),
            ((1, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 2, 9)], 0), True),
            ((1, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 9)], 0), True),
            ((1, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), -2, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -3, 9)], 0), True),
            ((1, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), -2, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), 0, 9)], 0), True),
            ((1, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), -2, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 4, 9)], 0), True),
            ((1, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), -2, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 2, 9)], 0), True),
            ((1, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), -2, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 9)], 0), True),
            ((1, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), 0, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), 0, 9)], 0), True),
            ((1, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), 0, 4), ((0, 0), (5, 6), (1, 1), 2, 9)], 0), True),
            # ((1, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), 0, 4), ((0, 0), (5, 6), (1, 1), 2, 9)], 0), False), need to handle introduction of new variables
            ((1, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), 0, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 2, 9)], 0), True),
            ((1, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), 0, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 9)], 0), True),
            ((1, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 3, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 4, 9)], 0), True),
            ((1, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 3, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 2, 9)], 0), True),
            ((1, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 3, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 9)], 0), True),
            # ((1, 1, [((0, 0, 0, 0), (1, 2, 3, 4), (1, 1, 1, 1), 2, 9), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 2, 10)], 0), True),
            # ((1, 1, [((0, 0, 0, 0), (1, 2, 3, 4), (1, 1, 1, 1), 2, 9), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 10)], 0), True),
            # ((1, 1, [((0, 0, 0, 0), (1, 2, 3, 4), (-1, -1, -1, -1), -2, 9), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 10)], 0), True),
            ((1, 1, [
                ((0,), (1,), (1,), 1, 5),
                ((0,), (2,), (1,), 1, 6),
                ((0,), (3,), (1,), 1, 7),
                ((0,), (4,), (1,), 1, 8)], 0), True),
            ((2, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0), (5, 6), (1, 1), 1, 7)], 0), True),
            #2 ((2, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -3, 9)], 0), True),
            ((2, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), 0, 9)], 0), True),
            ((2, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 4, 9)], 0), True),
            ((2, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 2, 9)], 0), True),
            ((2, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 1, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 9)], 0), True),
            #7 ((2, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), -2, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -3, 9)], 0), True),
            ((2, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), -2, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), 0, 9)], 0), True),
            ((2, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), -2, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 4, 9)], 0), True),
            #10 ((2, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), -2, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 2, 9)], 0), True),
            #11 ((2, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), -2, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 9)], 0), True),
            ((2, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), 0, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), 0, 9)], 0), True),
            ((2, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), 0, 4), ((0, 0), (5, 6), (1, 1), 2, 9)], 0), True),
            ((2, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), 0, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 2, 9)], 0), True),
            ((2, 1, [((0, 0, 0), (1, 2, 3), (-1, -1, -1), 0, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 9)], 0), True),
            ((2, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 3, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 4, 9)], 0), True),
            ((2, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 3, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 2, 9)], 0), True),
            ((2, 1, [((0, 0, 0), (1, 2, 3), (1, 1, 1), 3, 4), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 9)], 0), True),
            #19 ((2, 1, [((0, 0, 0, 0), (1, 2, 3, 4), (1, 1, 1, 1), 2, 9), ((0, 0, 0, 0), (5, 6, 7, 8), (1, 1, 1, 1), 2, 10)], 0), True),
            # ((2, 1, [((0, 0, 0, 0), (1, 2, 3, 4), (1, 1, 1, 1), 2, 9), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 10)], 0), True),
            # ((2, 1, [((0, 0, 0, 0), (1, 2, 3, 4), (-1, -1, -1, -1), -2, 9), ((0, 0, 0, 0), (5, 6, 7, 8), (-1, -1, -1, -1), -2, 10)], 0), True),
            ((4, 1, [
                ((0,), (1,), (1,), 1, 5),
                ((0,), (2,), (1,), 1, 6),
                ((0,), (3,), (1,), 1, 7),
                ((0,), (4,), (1,), 1, 8)], 0), True),
        
    ]

    for constraint in constraints:
        original_cc = puan.logic.plog._CompoundConstraint(*constraint[0])
        indices = tuple(set(itertools.chain.from_iterable(map(lambda x: x.index, original_cc.constraints))))
        number_of_variables = len(indices)
        values = numpy.array(list(map(lambda x: numpy.tile(numpy.append(numpy.zeros(int(math.pow(2,x))),
                                                                        numpy.ones(int(math.pow(2,x)))), int(math.pow(2,number_of_variables-1-x))),
                                                                        range(number_of_variables))), dtype=numpy.int64).T
        states = list(zip(itertools.repeat(indices, len(values)), map(tuple,values)))
        for state in states:
            a = original_cc.satisfied(state)
            b = puan.logic.plog._CompoundConstraint(*lf.transform(*constraint)).satisfied(state)
            assert a==b



def test_our_simplex():
    test_cases = [
        #(
        #   input    
        #   (
        #       polyhedron,
        #       objective_function
        #   ),
        #   expected_output
        #   (
        #       z*, value of objective function at optimum solution
        #       x*, value of variables at optimum solution
        #       A*, pivoted matrix A at optimum solution
        #       solution_information
        #   )
        # ),
        (
            # Input
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-3600, -7, -10], [-5400, -16, -12]])), 
                [(0, numpy.inf), (0, numpy.inf)],
                numpy.array([20, 18])
            ),
            # Solution with fractional numbers
            (   
                7531.57894736842,
                numpy.array([142.10526316, 260.52631579]),
                numpy.array([
                    [ 0.00000000e+00,  1.00000000e+00,  2.10526316e-01, -9.21052632e-02],
                    [ 1.00000000e+00, -1.11022302e-16, -1.57894737e-01, 1.31578947e-01]
                ]),
                'Solution is unique'
            )
        ),
        (
            # Input
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-80000, -30, -30, -30], [-80000, -25, -50, -45], [-57000, -30, 0, 0], [-84000, 0, -60, -50]])), 
                [(0, numpy.inf), (0, numpy.inf), (0, numpy.inf)],
                numpy.array([12000, 19000, 18000])
            ),
            # Expected output
            (
                35800000.00000001,
                numpy.array([1900, 0, 722.22]),
                numpy.array([
                    [ 1.22124533e-15, -3.33333333e+00, -4.24351912e-15, 1.00000000e+00, -6.66666667e-01, -4.44444444e-01, -1.82570008e-17],
                    [ 1.00000000e+00,  2.40548322e-18,  1.15648232e-17, 0.00000000e+00,  2.86807615e-18,  3.33333333e-02, -2.34997207e-18],
                    [-4.44089210e-16,  4.44444444e+00,  7.10542736e-15, 0.00000000e+00, -1.11111111e+00,  9.25925926e-01, 1.00000000e+00],
                    [-5.55111512e-17,  1.11111111e+00,  1.00000000e+00, 0.00000000e+00,  2.22222222e-02, -1.85185185e-02, 0.00000000e+00]
                ]),
                'Solution is unique'
            )
        ),
        (
            # Origo is not a BFS, phase I and II required to find a solution
            (
                puan.ndarray.ge_polyhedron(numpy.array([[1, 2, -1, -1], [-5, -2, -1, 0], [1, 0, -1, 1], [-1, 0, 1, -1]])),
                [(0, numpy.inf), (0, numpy.inf), (0, numpy.inf)],
                numpy.array([-2, -1, -1])
            ),
            (
                -3,
                numpy.array([1, 0, 1]),
                numpy.array([
                    [ 1. , -1. ,  0. , -0.5,  0. , -0.5,  0.],
                    [ 0. ,  3. ,  0. ,  1. ,  1. ,  1. ,  0.],
                    [ 0. , -1. ,  1. ,  0. ,  0. , -1. ,  0.],
                    [ 0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  1.]
                ]),
                'Solution is unique'
            )
        ),
        (
            # Linear program with degenerate solutions along the solving path
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-2, -1, 1], [-4, -2, 1], [-2, 0, -1]])), 
                [(0, numpy.inf), (0, numpy.inf)],
                numpy.array([1, 1])
            ),
            (
                5,
                numpy.array([3, 2]),
                numpy.array([
                    [ 1. ,  0. ,  0. ,  0.5,  0.5],
                    [ 0. ,  1. ,  0. ,  0. ,  1. ],
                    [ 0. ,  0. ,  1. , -0.5,  0.5]
                ]),
                'Solution is unique'
            )
        ),
        (
            # Another problem with degenerate solutions along the solving path
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-8, -1, -1, 0], [0, 0, 1, -1]])), 
                [(0, numpy.inf), (0, numpy.inf), (0, numpy.inf)],
                numpy.array([1, 1, 1])
            ),
            (
                16,
                numpy.array([0, 8, 8]),
                numpy.array([
                    [1., 1., 0., 1., 0.],
                    [1., 0., 1., 1., 1.]
                ]),
                'Solution is unique'
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-1, -1, -1]])),
                [(0, numpy.inf), (0, numpy.inf)],
                numpy.array([1, 1])
            ),
            (
                1,
                numpy.array([1, 0]),
                puan.ndarray.integer_ndarray([[1., 1., 1.]]),
                'Solution is not unique'
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-100, -2, -1], [-80, -1, -1]])),
                [(0, 40), (0, numpy.inf)], 
                numpy.array([30, 20])
            ),
            (
                1800,
                numpy.array([20, 60]),
                numpy.array([
                    [ 0.,  1., -1.,  2.],
                    [ 1.,  0., -1.,  1.]
                ]),
                'Solution is unique'
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-30, -2, -1]])),
                [(0, 10), (0, 15)],
                None,
                [numpy.array([3, 1]), numpy.array([1, 1])]
            ),
            (
                [
                    (
                        40.0,
                        numpy.array([10., 10.]),
                        numpy.array([[ -2.,  1., 1.]]),
                        'Solution is unique'
                    ),
                    (
                        22.5,
                        numpy.array([ 7.5, 15. ]),
                        numpy.array([[ 1. ,  0.5, -0.5]]),
                        'Solution is unique'
                    )
                ]
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-100, -2, -1], [-80, -1, -1], [-40, -1, 0]])),
                [(0, numpy.inf, int), (0, numpy.inf, int)],
                numpy.array([30, 20]),
            ),
            (
                1800,
                numpy.array([20, 60]),
                numpy.array([
                    [ 0.,  1., -1.,  2.,  0.],
                    [ 0.,  0., -1.,  1.,  1.],
                    [ 1.,  0.,  1., -1.,  0.]
                ]),
                'Solution is unique'
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-22, -11, 8], [0, 11, -12]])),
                [(0, numpy.inf, int), (0, numpy.inf, int)],
                numpy.array([1, 1]),
            ),
            (
                7,
                numpy.array([4, 3]),
                numpy.array([[-11, 8, 1, 0],
                             [11, -12, 0, 1]]),
                'Solution is unique'
            )
        ),
        (

            (
                puan.ndarray.ge_polyhedron(numpy.array([[-3, 1, -1], [-4, 0, 1], [-8, -1, -1], [-5, -1, 1], [3, 1, 1], [-12, 1, -4]])),
                [(0, numpy.inf), (0, numpy.inf)],
                numpy.array([-1, 1])
            ),
            (
                3,
                numpy.array([0, 3]),
                numpy.array([[-2, 0, 1, 0, 0, 0, 1, 0],
                             [-1, 0, 1, 1, 0, 0, 0, 0],
                             [2, 0, -1, 0, 1, 0, 0, 0],
                             [0, 0, 1, 0, 0, 1, 0, 0],
                             [-1, 1, 1, 0, 0, 0, 0, 0],
                             [3, 0, -4, 0, 0, 0, 0, 1]]),
                'Solution is not unique'
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-6, -2, -2, 1], [-5, -1, -3, 1], [-10, -2, -1, -4]])),
                [(0, numpy.inf), (0, numpy.inf), (0, numpy.inf)],
                numpy.array([4, -2, 3])
            ),
            (
                16,
                numpy.array([17/5, 0, 4/5]),
                numpy.array([[1, 0.9, 0, 0.4, 0, 0.1],
                    [5.55e-17, 1.9, 0, -0.6, 1, 0.1],
                    [0, -0.2, 1, -0.2, 0, 0.2]]),
                'Solution is unique'
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-8, -4, -2, -1], [-10, -3, -2, -2], [-4, -1, -1, -1]])),
                [(0, numpy.inf), (0, numpy.inf), (0, numpy.inf)],
                numpy.array([5, 2, 1])
            ),
            (
                10,
                numpy.array([2, 0, 0]),
                numpy.array([[1, 0.5, 0.25, 0.25, 0, 0],
                    [0, 0.5, 1.25, -0.75, 1, 0],
                    [0, 0.5, 0.75, -0.25, 0, 1]]),
                'Solution is unique' 
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-9, -2, -3, 1], [4, 0, 2, 1], [6, 1, 0, 1], [-6, -1, 0, -1]])),
                [(0, numpy.inf, int), (0, numpy.inf, int), (0, numpy.inf, int)],
                numpy.array([2, 1, 1]),
            ),
            (
                11,
                numpy.array([0, 5, 6]), # or [1, 4, 5]
                numpy.array([[1, -1, 5.55e-17, 0.333, 0, 0, 0.333],
                             [0, 3, -2.22e-16, -0.333, 1, 0, 0.667],
                             [0, 1, 1, -0.333, 0, 0, 0.667],
                             [0, 0, 0, 0, 0, 1, 1]]),
                'Solution is not unique'
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-10, -2, -2, -1], [5, 1, 0, 2]])),
                [(0, numpy.inf), (0, numpy.inf), (0, numpy.inf)],
                numpy.array([-1 , 1, -1])
            ),
            (
                1.25,
                numpy.array([0, 3.75, 2.5]),
                numpy.array([[0.75, 1, 0, 0.5, 0.25],
                             [0.5, 0, 1, 0, -0.5]]),
                'Solution is unique'
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-4, -4, 1, -1, 0], [-1, -1, -1, 0, 1], [-4, 0, -1, -1, -1]])),
                [(0, numpy.inf), (0, numpy.inf), (0, numpy.inf), (0, numpy.inf)],
                numpy.array([4, 2, 3, 1])
            ),
            (
                12.666666666666668,
                numpy.array([1/3, 2/3, 10/3, 0]),
                numpy.array([[1, 0, 0, -1/2, 1/6, 1/3, -1/6], [0, 1, 0, -1/2, -1/6, 2/3, 1/6], [0, 0, 1, 3/2, 1/6, -2/3, 5/6]]),
                'Solution is unique' 
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-21, -5, -4]])),
                [(0, numpy.inf, int), (0, numpy.inf, int)],
                numpy.array([5, 2]),
            ),
            (
                20,
                numpy.array([4, 0]),
                numpy.array([[-5, -4, 1]]),
                'Solution is unique'
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-7, -4, -2, -2, -3]])),
                [(0, 1, int), (0, 1, int), (0, 1, int), (0, 1, int)],
                numpy.array([7, 3, 2, 2]),
            ),
            (
                10,
                numpy.array([1,1,0,0]),
                numpy.array([[-4, -2, -2, -3, 1]]),
                'Solution is unique'
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron(numpy.array([[4, 1, 4], [6, 3, 2]])),
                [(0, numpy.inf, int), (0, numpy.inf, int)],
                numpy.array([-4, -5]),
            ),
            (
                -13,
                numpy.array([2,1]),
                numpy.array([[0, 1, 0, 0, 0, -1],
                             [1, 0, 0, 0, -1, 0],
                             [0, 0, 0, 1, -3, -2],
                             [0, 0, 1, 0, -1, -4]]),
                'Solution is unique'
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-6, 4, -3], [-18, -3, -2]])),
                [(0, numpy.inf, int), (0, numpy.inf, int)],
                numpy.array([1, 5]),
            ),
            (
                23,
                numpy.array([3, 4]),
                numpy.array([[4, -3, 1, 0, 0],
                             [-3, -2, 0, 1, 0],
                             [1, 0, 0, 0, 1]]),
                'Solution is unique'
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-35, -5, -7], [-3, -1, 0], [-4, 0, -1]])),
                [(0, numpy.inf, int), (0, numpy.inf, int)],
                numpy.array([5, 6]),
            ),
            (
                29,
                numpy.array([1, 4]),
                numpy.array([[-5, 0, 1, 0, -7, 0],
                             [-1, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 1],
                             [0, 1, 0, 0, 1, 0]]),
                'Solution is unique'
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-7, -4, -2, -3, -1]])),
                [(0, numpy.inf, int), (0, numpy.inf, int), (0, numpy.inf, int), (0, numpy.inf, int)],
                numpy.array([18, 8, 4, 2]),
            ),
            (
                28,
                numpy.array([1,1,0,1]),
                numpy.array([[-4, -2, 3, 1, 1]]),
                'Solution is unique'

            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-8, 1, -1, -3], [-10, -3, -2, 1]])),
                [(0, numpy.inf, int), (0, numpy.inf, int), (0, numpy.inf, int)],
                numpy.array([-2, 3, 4]),
            ),
            (
                19,
                numpy.array([0, 5, 1]),
                numpy.array([[-0.333, -0.333, 1, 0.333, 0],
                             [2.67, -2.33, 0, 0.333, 1]]),
                'Solution is unique'
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-36, -6, -10, -15, -10]])),
                [(0, numpy.inf, int), (0, numpy.inf, int), (0, numpy.inf, int), (0, numpy.inf, int)],
                numpy.array([5, 10, 10, 16]),
            ),
            (
                53,
                numpy.array([1, 0, 0, 3]),
                numpy.array([[1, -1.67, 2.5, -1.67, 0.167]]),
                'Solution is unique'
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-9, -2, -3, -3]])),
                [(0, 2, int), (0, 2, int), (0, 2, int)],
                numpy.array([5, 6, 7]),
            ),
            (
                20,
                numpy.array([0, 1, 2]),
                numpy.array([[1, 0, 0, 0.5, 1.5, 1.5],
                             [0, 1, 0, 0, -1, 0],
                             [0, 0, 1, 0, 0, -1]]),
                'Solution is unique'
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-8, -2, -4, -2, -3, 0], [4, 2, 0, -1, 1, 3], [-6, -2, -3, 0, -1, 1], [-6, -3, -4, 0, -3, -1]])),
                [(0, 1, int), (0, 1, int), (0, 1, int), (0, 1, float), (0, 1, float)],
                numpy.array([4, 10, 2, 6, 1]),
            ),
            (
                11.25,
                numpy.array([1, 0, 1, 0.75, 0.75]),
                numpy. array([[ 0.,  0.5, -1.625, 0.,  0.,  1., -0.375,  0., -1.125, -0.625],
                              [ 0.,  0.5,  0.375, 0.,  1.,  0., -0.375,  0., -0.125, 0.375],
                              [ 0., -1.,  0.5, 0.,  0.,  0., -0.5,  1., -0.5, 1.5],
                              [ 0., -1.5, -0.125, 1.,  0.,  0., 0.125,  0.,  0.375, 0.875],
                              [ 1.,  0.,  0., 0.,  0.,  0., 0.,  0.,  0., -1.]]),
                'Solution is unique'
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-2, -1, -2, -1, 0, 0], [-13, -2, -4, -3, -7, -3]])),
                [(0, 1, int), (0, 1, int), (0, 1, int), (0, 1, int), (0, 1, int)],
                numpy.array([5, 7, 6, 4, 5]),
            ),
            (
                16,
                numpy.array([1, 0, 1, 0, 1]),
                numpy.array([[1, 2, -1, 0, 0, 1, 0],
                             [0, 0, -1, -7, -3, -2, 1]]),
                'Solution is unique'
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron(numpy.array([[-4, -2, -1, -2, -1]])),
                [(0, 1, int), (0, 1, int), (0, 1, int), (0, 1, int)],
                numpy.array([5, 8, 4, 6]),
            ),
            (
                19,
                numpy.array([1, 1, 0, 1]),
                numpy.array([[1, -0.5, 1, -0.5, 0.5]]),
                'Solution is unique'
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron([[-6, -1, -1, -1, -2], [-6, -3, -1, 0, 0], [-2, 1, -1, 0, 0], [-2, 0, 0, -1, 1], [-3, 0, 0, -1, -1]]),
                [(0, numpy.inf), (0, numpy.inf), (0, numpy.inf), (0, numpy.inf)],
                numpy.array([1, 4, 1, 2])
            ),
            (
                15,
                numpy.array([1, 3, 0, 1]),
                numpy.array([[0, 0, 0.5, 1, 0.5, -0.25, -0.25, 0, 0],
                             [1, 0, 0, 0, 0, 0.25, -0.25, 0, 0],
                             [0, 1, 0, 0, 0, 0.25, 0.75, 0, 0],
                             [0, 0, 1.5, 0, 0.5, -0.25, -0.25, 1, 0],
                             [0, 0, 0.5, 0, -0.5, 0.25, 0.25, 0, 1]]),
                'Solution is not unique'
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron([[-8, -1, -3, -2], [5, 1, 2, 1], [-4, -1, -1, 0]]),
                [(0, 1, int), (0, 2, int), (0, 2, int)],
                numpy.array([-2, -3, -3]),
            ),
            (
                -8,
                numpy.array([1, 2, 0]),
                numpy.array([[0, -1, 1, 1, 1, 0],
                             [1, -2, 1, 0, -1, 0],
                             [0, 1, -1, 0, 1, 1]]),
                'Solution is unique' 
            )
        ),
        (
            (
                puan.ndarray.ge_polyhedron([[5, 3, 1, -2, 2], [8, -1, 3, 2, 2], [-4, -1, 0, 0, -1], [-1, 0, -1, 0, 0], [-2, 0, 0, -1, 0]]),
                [(0, numpy.inf, int), (0, numpy.inf, int), (0, numpy.inf, int), (0, numpy.inf, int)],
                numpy.array([-1, -2, 1, -2]),
            ),
            (
                -5,
                numpy.array([1, 0, 2, 3]),
                numpy.array([[-1.5,  0.5,  0. ,  1. , -0.5,  0. ,  0. ,  0. ,  0. , -1. ],
                             [ 0. ,  0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -1. ],
                             [ 0.5, -0.5,  0. ,  0. ,  0.5,  0. ,  1. ,  0. ,  0. ,  1. ],
                             [ 0. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  0. ,  0. ],
                             [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  1. ,  1. ],
                             [-4. , -2. ,  0. ,  0. , -1. ,  1. ,  0. ,  0. ,  0. , -4. ]]),
                'Solution is unique'  
            )
        ),
        (
            (
                #                                    X11  X12, x21, x22, x31, x32, y1, y2, I1, I2,  L
                puan.ndarray.ge_polyhedron([[-1800,  -8,   0, -10,   0,  -9,   0,  0,  0,  0,  0,  0],
                                             [-1800,   0,  -8,   0, -10,   0,  -9,  0,  0,  0,  0,  0],
                                             [-2400,  -7,   0,  -9,   0,  -8,   0,  0,  0,  0,  0,  0],
                                             [-2400,   0,  -7,   0,  -9,   0,  -8,  0,  0,  0,  0,  0],
                                             [-2100, -10,   0,  -8,   0, -11,   0,  0,  0,  0,  0,  0],
                                             [-2100,   0, -10,   0,  -8,   0, -11,  0,  0,  0,  0,  0],
                                             [    0,  75,  75, -25, -25, -25, -25,  0,  0,  0,  0,  0],
                                             [  -40,  -3,   0,  -6,   0,  -4,   0,  1,  0, -1,  0, -1],
                                             [   40,   3,   0,   6,   0,   4,   0, -1,  0,  1,  0,  1],
                                             [    0,   0,  -3,   0,  -6,   0,  -4,  0,  1,  1, -1,  0],
                                             [    0,   0,   3,   0,   6,   0,   4,  0, -1, -1,  1,  0]]),
                [(0, numpy.inf), (0, numpy.inf), (0, numpy.inf), (0, numpy.inf), (0, numpy.inf), (0, numpy.inf), (0, numpy.inf), (0, 400), (0, 200), (0, 200), (0, numpy.inf)],
                numpy.array([190, 190, 240, 240, 210, 210, -5, -8, -2, -2, -4])
            ),
            (
                72683.33333333336,
                numpy.array([183.333, 200, 33.3333, 0, 0, 0, 910, 400, 200, 0, 0]),
                numpy.array([[ 0., 0.,  1., 0.,  5.55555556e-02, 0., 0., 0., 0., 0.,  0.,  2.77777778e-01, 0.,  0.,  0., -2.22222222e-01,  0.,  0, 0.,  0., 0., 0.],
                             [ 0., 0.,  0., -6.,  0., -1.66666667, 0., -2.66666667, -2.66666667, -2.66666667,  0.,  0., 1.,  0.,  0., 0.,  0.,  0., 0.,  0., -2.66666667, 0.],
                             [ 0., 0.,  0., 0.,  1.11111111e-01, 0., 0., 0., 0., 0.,  0., -9.44444444e-01, 0.,  1.,  0., 5.55555556e-02,  0.,  0., 0.,  0., 0., 0.],
                             [ 0., 0.,  0., -5.,  0., -1.33333333, 0., -2.33333333, -2.33333333, -2.33333333,  0.,  0., 0.,  0.,  1., 0.,  0.,  0., 0.,  0., -2.33333333, 0.],
                             [ 0., 0.,  0., 0., -5.00000000e-01, 0., 1., 0.,  1., 0., -1.,  1., 0.,  0.,  0., -5.00000000e-01,  0.,  0., -1.,  0., 0., 0.],
                             [ 0.,  0.,  0., -1.20000000e+01,  0., -2.33333333, 0., -3.33333333, -3.33333333, -3.33333333,  0.,  0., 0.,  0.,  0., 0.,  1.,  0., 0.,  0., -3.33333333, 0.],
                             [ 1., 0.,  0., 0.,  1.05555556, 0., 0., 0., 0., 0.,  0., -2.22222222e-01, 0.,  0.,  0., 2.77777778e-01,  0.,  0., 0.,  0., 0.,0.],
                             [ 0.,  0.,  0., 0.,  0.,  0., 0.,  0.,  0., 0.,  0.,  0., 0.,  0.,  0., 0.,  0.,  0., 1.,  1.,  0., 0.],
                             [ 0.,  1.,  0., 2.,  0.,  1.33333333, 0.,  3.33333333e-01,  3.33333333e-01, 3.33333333e-01,  0.,  0., 0.,  0.,  0., 0.,  0.,  0., 0.,  0.,  3.33333333e-01, 0.],
                             [ 0.,  0.,  0., 1.75e+02,  1.02777778e+02,  1.25e+02, 0.,  2.5e+01,  2.5e+01, 2.5e+01,  0., -2.36111111e+01, 0.,  0.,  0., 2.63888889e+01,  0.,  1., 0.,  0.,  2.5e+01, 0.],
                             [ 0.,  0.,  0., 0.,  0.,  0., 0.,  0.,  0., 0.,  0.,  0., 0.,  0.,  0., 0.,  0.,  0., 0.,  0.,  1., 1.]]),
                'Solution is unique'
            )
        )

    ]

    for (input, expected_output) in test_cases:
        actual = puan.solver.our_revised_simplex(*input)
        if type(expected_output) == list:
            for ao, eo in zip(actual, expected_output):
                assert ao[0] == eo[0]
                assert numpy.allclose(ao[1], eo[1])
                assert numpy.allclose(ao[2], eo[2])
                assert ao[3] == eo[3]
        else:
            assert abs(actual[0] - expected_output[0]) < 0.01
            assert numpy.allclose(actual[1], expected_output[1])
            assert numpy.allclose(actual[2], expected_output[2], rtol=0.000001, atol=0.01)
            assert actual[3] == expected_output[3]
    
    # Unbounded problem
    objective_function = numpy.array([1, 1])
    polyhedron = puan.ndarray.ge_polyhedron(numpy.array([[-3, 0, -1]]))
    bounds = [(0, numpy.inf), (0, numpy.inf)]
    expected = (numpy.inf, numpy.array([numpy.inf, 0]), 'Solution is unbounded')
    actual = puan.solver.our_revised_simplex(polyhedron, bounds, objective_function)
    assert actual[0] == expected[0]
    assert numpy.allclose(actual[1], expected[1])
    assert actual[3] == expected[2]

    # No feasible solution
    objective_function = numpy.array([1, 1])
    polyhedron = puan.ndarray.ge_polyhedron(numpy.array([[-1, -1, -1], [0, 1, -1], [2, 0, 1]])) 
    bounds=[(0, numpy.inf), (0, numpy.inf)]
    expected = (None, None, None, "No feasible solution exists")
    actual = puan.solver.our_revised_simplex(polyhedron, bounds, objective_function)
    assert actual == expected