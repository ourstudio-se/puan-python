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
import numpy
import operator
import maz
import math

from hypothesis import example, given, strategies as st, settings, assume

def atom_proposition_strategy():
    return st.builds(
        puan.variable,
        id=st.text(), 
        bounds=st.tuples(
            st.integers(min_value=-10, max_value=10),
            st.integers(min_value=-10, max_value=10),
        ),
    )

def proposition_strategy():
    return st.builds(
        pg.AtLeast, 
        propositions=st.iterables(
            atom_proposition_strategy(),
            max_size=5
        ),
        variable=st.text(), 
        value=st.integers(min_value=-5, max_value=5),
    )

@given(proposition_strategy())
def test_model_variables_hypothesis(model):
    _ = model.propositions

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
    variables = list(set(map(operator.attrgetter("id"), model.flatten())))
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
    conj_props = pg.All(*map(pg.Imply.from_cicJE, rules), variable="A")
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
    assert first_prop.propositions[0].sign == -1
    assert len(first_prop.propositions[0].propositions) == 6
    assert first_prop.propositions[1].sign == 1
    assert len(first_prop.propositions[1].propositions) == 3

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
            puan.variable("0"),
            puan.variable("a"),
            puan.variable("b"),
            puan.variable("c"),
            puan.variable("d")
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

    a_req_b = pg.Imply("a","b", variable="a_req_b").to_dict()
    b_req_c = pg.Imply("b","c", variable="b_req_c").to_dict()

    for k,v in a_req_b.items():
        assert (not k in b_req_c) or b_req_c[k] == v, f"{k} is ambivalent: means both {v} and {b_req_c[k]}"

def test_bind_relations_to_compound_id():

    model = pg.All(
        pg.Any("a","b",variable="B"),
        pg.Any("c","d",variable="C"),
        pg.Any("C","B",variable="A"),
        variable="model"
    )
    assert len(model.propositions) == 3

def test_dont_override_propositions():

    model = pg.All(
        pg.Imply(
            condition=pg.AtLeast(
                value=3, 
                propositions=[
                    puan.variable("t", bounds=(0,10))
                ]
            ), 
            consequence=pg.All(*"abc"), 
            variable=puan.variable("xor_xy", bounds=(0,50)),
        ),
        pg.Xor("x","y", variable="xor_xy"),
    )
    assert model.propositions[0].variable.bounds != model.propositions[1].variable.bounds

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

def test_xnor_proposition():
    
    actual_model = pg.XNor("x","y","z") # <- meaning none or at least 2
    polyhedron = actual_model.to_polyhedron(True)
    assert polyhedron.shape == (3,6)
    assert polyhedron[polyhedron.A.dot([0,1,1,1,1]) < polyhedron.b].size == 0
    assert polyhedron[polyhedron.A.dot([0,1,1,1,0]) < polyhedron.b].size == 0

def test_proposition_polyhedron_conversions():

    actual = pg.All(
        pg.Not("x"),
        variable="all_not"
    ).to_polyhedron(True)
    assert not all(actual.A.dot(actual.A.construct(*{"x": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.A.construct(*{"VARec31682fde561917952ff78a7a8adeffd0febc372dd26871916c46c630381b45": 1}.items())) >= actual.b)

    actual = pg.Not(
        pg.All("x","y","z", variable="all_xyz")
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
    assert all(actual.A.dot(actual.A.construct(*{"VARec31682fde561917952ff78a7a8adeffd0febc372dd26871916c46c630381b45": 1, "x": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.A.construct(*{"VAR851ca7a5e2d4bce908ced2c566ce1ef6f1cc1921fcb4c270353cbc81f2e3b59c": 1, "a": 1, "b": 1, "c": 1}.items())) >= actual.b)

    actual = pg.Not(
        pg.Imply(
            condition=pg.All("x","y","z"), 
            consequence=pg.Any("a","b","c")
        ),
    ).to_polyhedron(True)
    assert all(actual.A.dot(actual.A.construct(*{"VAR86fca5ee438a294cecf9003ee4f8c73dcf99f07273ba6d609ed49abe9e2d3d2d": 1, "VARdbfcfd0d87220f629339bd3adcf452d083fde3246625fb3a93e314f833e20d37": 1, "x": 1, "y": 1, "z": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"VAR86fca5ee438a294cecf9003ee4f8c73dcf99f07273ba6d609ed49abe9e2d3d2d": 1, "VARdbfcfd0d87220f629339bd3adcf452d083fde3246625fb3a93e314f833e20d37": 1, "x": 1, "y": 1, "z": 1, "a": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"VAR86fca5ee438a294cecf9003ee4f8c73dcf99f07273ba6d609ed49abe9e2d3d2d": 1, "VARdbfcfd0d87220f629339bd3adcf452d083fde3246625fb3a93e314f833e20d37": 1, "x": 1, "y": 1, "z": 1, "b": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"VAR86fca5ee438a294cecf9003ee4f8c73dcf99f07273ba6d609ed49abe9e2d3d2d": 1, "VARdbfcfd0d87220f629339bd3adcf452d083fde3246625fb3a93e314f833e20d37": 1, "x": 1, "y": 1, "z": 1, "c": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"VAR86fca5ee438a294cecf9003ee4f8c73dcf99f07273ba6d609ed49abe9e2d3d2d": 1, "VARdbfcfd0d87220f629339bd3adcf452d083fde3246625fb3a93e314f833e20d37": 1, "x": 1, "y": 1}.items())) >= actual.b)

    actual = pg.Not(
        pg.AtLeast(propositions=["x","y","z"], value=2)
    ).to_polyhedron(True)
    assert not all(actual.A.dot(actual.A.construct(*{"x": 1, "y": 1, "z": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.A.construct(*{"x": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.A.construct(*{"y": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.A.construct(*{"z": 1}.items())) >= actual.b)

    actual = pg.Not(
        pg.AtMost(propositions=["x","y","z"], value=2)
    ).to_polyhedron(True)
    assert not all(actual.A.dot(actual.A.construct(*{"x": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"y": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"z": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"x": 1, "y": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"x": 1, "z": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"y": 1, "z": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.A.construct(*{"x": 1, "y": 1, "z": 1}.items())) >= actual.b)

# def test_assuming_integer_variables():

#     """
#         Assumes x,y,z and we'll expect that t must be between
#         10 and 20.
#     """
#     model = pg.Imply(
#         pg.All(*"xyz"),
#         pg.All(
#             pg.AtLeast(
#                 pg.Integer("t"),
#                 value=10,
#                 variable="at-least-10"
#             ),
#             pg.AtMost(
#                 pg.Integer("t"),
#                 value=20,
#                 variable="at-most-20"
#             )
#         ), 
#     )

#     assumed = model.assume({"x": 1, "y": 1, "z": 1})[0].to_polyhedron(True)
#     assert all(assumed.A.dot(assumed.A.construct(*{"at-least-10": 1, "at-most-20": 1, "t": 10}.items())) >= assumed.b)
#     assert all(assumed.A.dot(assumed.A.construct(*{"at-least-10": 1, "at-most-20": 1, "t": 20}.items())) >= assumed.b)
#     assert not all(assumed.A.dot(assumed.A.construct(*{"t": 1}.items())) >= assumed.b)
#     assert not all(assumed.A.dot(assumed.A.construct(*{"t": -22}.items())) >= assumed.b)
#     assert not all(assumed.A.dot(assumed.A.construct(*{"t": 9}.items())) >= assumed.b)
#     assert not all(assumed.A.dot(assumed.A.construct(*{"t": 21}.items())) >= assumed.b)

#     model = pg.Imply(
#         pg.All(
#             pg.AtLeast(
#                 pg.Integer("t"),
#                 value=10,
#                 variable="at-least-10"
#             ),
#             pg.AtMost(
#                 pg.Integer("t"),
#                 value=20,
#                 variable="at-most-20"
#             )
#         ), 
#         pg.All(*"xyz", variable="all-xyz")
#     )

#     assumed_model, assumed_items = model.assume({"t": 12})
#     assert assumed_model.is_tautologi
#     assert assumed_items['t'] == 12
#     assert assumed_items['x'] == 1
#     assert assumed_items['y'] == 1
#     assert assumed_items['z'] == 1

def test_bound_approx():

    # Test random constraints with a being an integer
    variables = [
        puan.variable("0", (-10,10)),
        puan.variable("a", (-10,10)),
        puan.variable("b", (0,1)),
        puan.variable("c", (0,1)),
        puan.variable("d", (0,1)),
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
        puan.variable("0", (-10,10)),
        puan.variable("a", (-10,10)),
        puan.variable("b", (0,1)),
        puan.variable("c", (0,1)),
        puan.variable("d", (0,1)),
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
        puan.variable("0", (-10,10)),
        puan.variable("a", (-10,10)),
        puan.variable("b", (0,1)),
        puan.variable("c", (0,1)),
        puan.variable("d", (0,1)),
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
        puan.variable("0", (-10,10)),
        puan.variable("a", (-10,10)),
        puan.variable("b", (-10,10)),
        puan.variable("c", (-10,10)),
        puan.variable("d", (-10,10)),
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
        puan.variable("0", (-10,10)),
        puan.variable("a", (-10,10)),
        puan.variable("b", (0,1))
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
        puan.variable("0", (-10,10)),
        puan.variable("a", (-10,10)),
        puan.variable("b", (0,1)),
        puan.variable("c", (-10,10)),
        puan.variable("d", (0,1)),
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
            "proposition": [
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
        puan.SolutionVariable("a",value=1.0),
        puan.SolutionVariable("b",value=1.0),
        puan.SolutionVariable("p",value=1.0),
        puan.SolutionVariable("q",value=1.0),
        puan.SolutionVariable("r",value=1.0),
        puan.SolutionVariable("x",value=1.0),
        puan.SolutionVariable("y",value=1.0),
        puan.SolutionVariable("z",value=1.0),
    ]
    actual = list(model.select({"a": 1}, solver=dummy_solver, only_leafs=True))
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

def test_at_least_negate():

    var = ["x","y","z"]
    actual = pg.AtLeast(0, var, variable="A").negate()
    expected = pg.AtLeast(1, var, variable="A", sign=-1)
    assert actual == expected

    actual = pg.AtLeast(1, var, variable="A").negate()
    expected = pg.AtLeast(0, var, variable="A", sign=-1)
    assert actual == expected

    actual = pg.AtLeast(2, var, variable="A").negate()
    expected = pg.AtLeast(-1, var, variable="A", sign=-1)
    assert actual == expected

    actual = pg.AtLeast(3, var, variable="A").negate()
    expected = pg.AtLeast(-2, var, variable="A", sign=-1)
    assert actual == expected

    actual = pg.AtLeast(-1, var, variable="A", sign=-1).negate()
    expected = pg.AtLeast(2, var, variable="A")
    assert actual == expected

    actual = pg.AtLeast(-2, var, variable="A", sign=-1).negate()
    expected = pg.AtLeast(3, var, variable="A")
    assert actual == expected

    actual = pg.AtLeast(-3, var, variable="A", sign=-1).negate()
    expected = pg.AtLeast(4, var, variable="A")
    assert actual == expected

    var = [puan.variable("x",(0,2)), "y", "z"]
    actual = pg.AtLeast(0, var, variable="A").negate()
    expected = pg.AtLeast(1, var, variable="A", sign=-1)
    assert actual == expected

    actual = pg.AtLeast(1, var, variable="A").negate()
    expected = pg.AtLeast(0, var, variable="A", sign=-1)
    assert actual == expected

    actual = pg.AtLeast(2, var, variable="A").negate()
    expected = pg.AtLeast(-1, var, variable="A", sign=-1)
    assert actual == expected

    actual = pg.AtLeast(3, var, variable="A").negate()
    expected = pg.AtLeast(-2, var, variable="A", sign=-1)
    assert actual == expected

    actual = pg.AtLeast(-1, var, variable="A", sign=-1).negate()
    expected = pg.AtLeast(2, var, variable="A")
    assert actual == expected

    actual = pg.AtLeast(-2, var, variable="A", sign=-1).negate()
    expected = pg.AtLeast(3, var, variable="A")
    assert actual == expected

    actual = pg.AtLeast(-3, var, variable="A", sign=-1).negate()
    expected = pg.AtLeast(4, var, variable="A")
    assert actual == expected

def test_proposition_negation():
    
    # Easy conversion is to negate one Any with two All's
    # Should en in an outer All and two AtMost 2
    model = pg.Any(
        pg.All(*"abc"),
        pg.All(*"xyz"),
    )
    negated = model.negate()
    assert len(negated.propositions) == 2
    assert negated.sign == 1
    assert negated.value == 2
    assert negated.propositions[0].sign == -1
    assert negated.propositions[0].value == -2
    assert negated.propositions[1].sign == -1
    assert negated.propositions[1].value == -2

    # Test that negate holds when atomic and composite
    # propositions are defined on same level
    model = pg.Any(
        pg.All(*"abc"),
        *"xyz",
    )
    negated = model.negate()
    assert len(negated.propositions) == 2
    assert negated.sign == 1
    assert negated.value == 2
    assert negated.propositions[0].sign == -1
    assert negated.propositions[0].value == -2
    assert negated.propositions[1].sign == -1
    assert negated.propositions[1].value == 0
    
    # Test to negate when having an AtMost
    # Expects to resolve into All(All(a,b,c),All(x,y,z))
    model = pg.AtMost(
        value=2,
        propositions=[
            pg.All(*"abc"),
            pg.All(*"xyz"),
        ]
    )
    negated = model.negate()
    assert len(negated.propositions) == 2
    assert negated.sign == 1
    assert negated.value == 3
    assert negated.propositions[0].sign == 1
    assert negated.propositions[0].value == 3
    assert negated.propositions[1].sign == 1
    assert negated.propositions[1].value == 3

    # Test to negate when AtMost inside which in turn
    # having positive compound propositions
    model = pg.All(
        pg.AtMost(
            value=1,
            propositions=[
                pg.Any(*"abc"),
                *"xyz"
            ]
        ),
        pg.Any(*"pqr")
    )
    negated = model.negate()
    assert len(negated.propositions) == 2
    assert negated.sign == 1
    assert negated.value == 1
    assert negated.propositions[0].sign == 1
    assert negated.propositions[0].value == 2
    assert negated.propositions[0].propositions[0].sign == 1
    assert negated.propositions[0].propositions[0].value == 1
    assert len(negated.propositions[0].propositions[1:]) == 3
    assert negated.propositions[1].sign == -1
    assert negated.propositions[1].value == 0

def test_configurator_to_json():

    expected = {
        'propositions': [
            {
                "type": "Xor",
                "propositions": [
                    {"id": "z"},
                    {"id": "x"},
                    {"id": "y"},
                ],
                "default": [
                    {"id": "x"}
                ]
            },
            {
                "type": "Any",
                "propositions": [
                    {"id": "b"},
                    {"id": "a"},
                    {"id": "c"},
                ],
                "default": [
                    {"id": "a"}
                ]
            },
            {
                "type": "Xor",
                "propositions": [
                    {"id": "p"},
                    {"id": "q"},
                ],
            },
            {
                "type": "Any",
                "propositions": [
                    {"id": "r"},
                    {"id": "s"},
                ],
            }
        ]
    }

    actual = cc.StingyConfigurator.from_json(expected).to_json()
    assert len(actual['propositions']) == len(expected['propositions'])
    for a,b in zip(actual['propositions'], expected['propositions']):
        assert a['type'] == b['type']
        assert len(a['propositions']) == len(b['propositions'])
        assert a.get('default', []) == b.get('default', [])