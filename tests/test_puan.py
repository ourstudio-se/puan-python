import ast
import functools
import puan
import puan.ndarray
import puan.vmap
import puan.logic
import puan.logic.plog as pg
import numpy
import operator
import maz

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
    rules_gen = puan.logic.sta.application.to_cicJEs(
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

def test_value_map2matrix():

    empty_value_map = {}
    matrix_one = puan.vmap.to_matrix(empty_value_map)
    matrix_two = puan.vmap.to_matrix(empty_value_map, m_max=1)
    assert matrix_one.shape == (1,1)
    assert matrix_two.shape == (1,2)

    empty_filled_value_map = {
        1: [[], []],
        -1: [[], []],
    }
    matrix_one = puan.vmap.to_matrix(empty_filled_value_map)
    matrix_two = puan.vmap.to_matrix(empty_filled_value_map, m_max=1)
    assert matrix_one.shape == (1, 1)
    assert matrix_two.shape == (1, 2)

    prio_value_map_test = {
         1: [[1,2,3],[3,2,1]],
        -1: [[1,2,3],[3,2,1]],
    }
    matrix_prio_test = puan.vmap.to_matrix(prio_value_map_test)
    assert (matrix_prio_test <= 0).all()

    diag_value_map = {
        2:  [[0], [0]],
        1:  [[1], [1]],
        -1: [[3], [3]],
        -2: [[4], [4]],
    }
    diag_mat = puan.vmap.to_matrix(diag_value_map)
    expected_diag_mat = numpy.array([
        [2,0,0,0,0],
        [0,1,0,0,0],
        [0,0,0,0,0],
        [0,0,0,-1,0],
        [0,0,0,0,-2],
    ])
    assert (diag_mat == expected_diag_mat).all()

    zero_replacing_value_map = {
        1: [[1], [1]],
        0: [[1], [1]]
    }
    zero_replacing_mat = puan.vmap.to_matrix(zero_replacing_value_map)
    expected_zero_replacing_mat = numpy.array([
        [0,0],
        [0,0]
    ])
    assert (zero_replacing_mat == expected_zero_replacing_mat).all()

def test_merge_value_maps():

    value_map_empty = {}
    value_map_zero = {
        2:  [[0], [0]],
        1:  [[1], [1]],
        -1: [[3], [3]],
        -2: [[4], [4]],
    }

    value_map_one = {
         1: [[1,2,3],[3,2,1]],
        -1: [[1,2,3],[3,2,1]],
    }

    merged_value_map_empty_left = puan.vmap.merge(value_map_empty, value_map_one)
    assert merged_value_map_empty_left == value_map_one

    merged_value_map_empty_right = puan.vmap.merge(value_map_one, value_map_empty)
    assert merged_value_map_empty_right == value_map_one

    merged_value_map_empty_both = puan.vmap.merge(value_map_empty, value_map_empty)
    assert merged_value_map_empty_both == {}

    merged_value_map = puan.vmap.merge(value_map_zero, value_map_one)
    assert merged_value_map == {
        2: [[0], [0]],
        1: [[1,6,7,8], [1,3,2,1]],
        -1: [[3,6,7,8], [3,3,2,1]],
        -2: [[4], [4]],
    }
    assert value_map_zero == {
        2:  [[0], [0]],
        1:  [[1], [1]],
        -1: [[3], [3]],
        -2: [[4], [4]],
    }
    assert value_map_one == {
         1: [[1,2,3],[3,2,1]],
        -1: [[1,2,3],[3,2,1]],
    }

def test_randomly_gen_value_maps_to_mats_and_back():

    """
        NOTE matrix -> value map -> matrix conversions
        should always hold. But value map -> matrix -> value map
        may in the cases where conflicts appear between cell values,
        not hold.

        NOTE also, a value map doesn't hold any information of zeros.
        Therefore, if any leading rows/columns are zeros, that information
        will not be stored in a value map. However, zeros in the middle/beginning
        of the matrix are implicitly defined by the other values around them and
        therefore they are included.
    """

    n_test_objects = 100
    for i in range(n_test_objects):

        rand_matrix = numpy.random.randint(-5,5,size=(numpy.random.randint(1,10), numpy.random.randint(1,10)))

        # for now, ship matrices with leading zeros
        zero_msk = rand_matrix == 0
        if zero_msk.T[-1].all(axis=0) or zero_msk[-1].all(axis=0):
            continue

        comparison = operator.eq(
            rand_matrix,
            puan.vmap.to_matrix(
                puan.ndarray.to_value_map(rand_matrix),
                dtype=rand_matrix.dtype,
            ),
        )
        assert comparison.all() if not isinstance(comparison, bool) else comparison

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

# def test_compress_rules():
#     rules = [
#         {'condition': {
#             'relation': "ALL",
#             'subConditions': [{
#                 'relation': "ALL",
#                 'components': [
#                     {'id': 'a'},
#                     {'id': 'b'}
#                     ]
#                 }
#             ]
#         },
#         'consequence': {
#             'ruleType': 'REQUIRES_ALL',
#             'components': [
#                 {'id': 'x'}
#                 ]
#             }
#         },
#         {'condition': {
#             'relation': "ALL",
#             'subConditions': [{
#                 'relation': "ALL",
#                 'components': [
#                     {'id': 'b'},
#                     {'id': 'a'}
#                     ]
#                 }]
#             },
#         'consequence': {
#             'ruleType': 'REQUIRES_ALL',
#             'components': [
#                 {'id': 'x'},
#                 {'id': 'y'}
#                 ]
#             }
#         },
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
#             'ruleType': 'FORBIDS_ALL',
#             'components': [
#                 {'id': 'x'},
#                 {'id': 'y'}
#                 ]
#             }
#         },
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
#             'ruleType': 'FORBIDS_ALL',
#             'components': [
#                 {'id': 'y'},
#                 {'id': 'z'}
#                 ]
#             }
#         },
#         {'condition': {
#             'relation': "ALL",
#             'subConditions': [{
#                 'relation': "ALL",
#                 'components': [
#                     {'id': 'a'},
#                     {'id': 'b'}
#                     ]
#                 }
#             ]
#         },
#         'consequence': {
#             'ruleType': 'ONE_OR_NONE',
#             'components': [
#                 {'id': 'x'},
#                 {'id': 'y'}
#                 ]
#             }
#         },
#         {'condition': {
#             'relation': "ALL",
#             'subConditions': [{
#                 'relation': "ALL",
#                 'components': [
#                     {'id': 'a'},
#                     {'id': 'b'}
#                     ]
#                 }
#             ]
#         },
#         'consequence': {
#             'ruleType': 'ONE_OR_NONE',
#             'components': [
#                 {'id': 'y'},
#                 {'id': 'z'}
#                 ]
#             }
#         },
#         {'condition': {
#             'relation': "ALL",
#             'subConditions': [{
#                 'relation': "ALL",
#                 'components': [
#                     {'id': 'a'},
#                     {'id': 'b'}
#                     ]
#                 }
#             ]
#         },
#         'consequence': {
#             'ruleType': 'REQUIRES_EXCLUSIVELY',
#             'components': [
#                 {'id': 'x'},
#                 {'id': 'y'}
#                 ]
#             }
#         },
#         {'condition': {
#             'relation': "ALL",
#             'subConditions': [{
#                 'relation': "ALL",
#                 'components': [
#                     {'id': 'a'},
#                     {'id': 'b'}
#                     ]
#                 }
#             ]
#         },
#         'consequence': {
#             'ruleType': 'REQUIRES_EXCLUSIVELY',
#             'components': [
#                 {'id': 'z'}
#                 ]
#             }
#         },
#         {'condition': {
#             'relation': "ALL",
#             'subConditions': [{
#                 'relation': "ALL",
#                 'components': [
#                     {'id': 'a'},
#                     {'id': 'b'}
#                     ]
#                 }
#             ]
#         },
#         'consequence': {
#             'ruleType': 'REQUIRES_ANY',
#             'components': [
#                 {'id': 'i'}
#                 ]
#             }
#         },
#         {'condition': {
#             'relation': "ALL",
#             'subConditions': [{
#                 'relation': "ALL",
#                 'components': [
#                     {'id': 'a'},
#                     {'id': 'b'}
#                     ]
#                 }
#             ]
#         },
#         'consequence': {
#             'ruleType': 'REQUIRES_ANY',
#             'components': [
#                 {'id': 'j'},
#                 {'id': 'k'}
#                 ]
#             }
#         }
#     ]

#     expected_result = [{
#         'condition': {
#         'relation': "ALL",
#         'subConditions': [{
#             'relation': "ALL",
#             'components': [
#                 {'id': 'a'},
#                 {'id': 'b'}
#                 ]
#             }]
#         },
#         'consequence': {
#             'ruleType': 'REQUIRES_ALL',
#             'components': [
#                 {'id': 'i'},  # From require any
#                 {'id': 'x'},
#                 {'id': 'y'},
#                 {'id': 'z'}  # From require exclusively
#                 ]
#             }
#         },
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
#             'ruleType': 'FORBIDS_ALL',
#             'components': [
#                 {'id': 'x'},
#                 {'id': 'y'},
#                 {'id': 'z'},
#                 ]
#             }
#         },
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
#             'ruleType': 'ONE_OR_NONE',
#             'components': [
#                 {'id': 'x'},
#                 {'id': 'y'}
#                 ]
#             }
#         },
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
#             'ruleType': 'ONE_OR_NONE',
#             'components': [
#                 {'id': 'y'},
#                 {'id': 'z'}
#                 ]
#             }
#         },
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
#             'ruleType': 'REQUIRES_EXCLUSIVELY',
#             'components': [
#                 {'id': 'x'},
#                 {'id': 'y'}
#                 ]
#             }
#         },
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
#         }
#     ]
#     actual = cc.cicJEs.compress(rules, id_ident="id")
#     for rule in actual:
#         rule['consequence']['components'] = sorted(rule['consequence']['components'], key=lambda d: d['id'])
#     assert [i for i in actual if i not in expected_result] == []
#     assert [i for i in expected_result if i not in actual] == []

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
    model = puan.logic.sta.application.to_cicJEs(
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
    assert isinstance(first_prop.propositions[0], pg.AtMost)
    assert len(first_prop.propositions[0].propositions) == 6
    assert isinstance(first_prop.propositions[1], pg.AtLeast)
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
    model = puan.logic.sta.application.to_cicJEs(
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
    result = list(puan.logic.sta.application.to_cicJEs(application, items))
    assert len(result) == 0


def test_reduce_matrix():

    matrix = numpy.array([
        [-1,-1,-1, 1, 0, 2], # stay
        [-2,-1,-1, 1, 0, 2], # remove
        [ 0,-1, 1, 1, 0, 0], # stay
        [-3,-2,-1,-1, 0, 0], # stay
        [-4,-1,-1,-1,-1,-1], # stay
        [ 2, 1, 1, 1, 1, 1], # stay
        [ 0, 1, 1, 0, 1, 0], # remove
        [ 0, 1, 1, 0, 1,-1], # stay
    ])
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
    expected = puan.ndarray.ge_polyhedron(numpy.array([-2, -2, -2]))
    assert numpy.array_equal(actual, expected)
    input = puan.ndarray.ge_polyhedron(numpy.array([[3, 1, 1, 1]]))
    actual = input.reducable_columns_approx()
    expected = puan.ndarray.ge_polyhedron(numpy.array([1, 1, 1]))
    assert numpy.array_equal(actual, expected)
    input = puan.ndarray.ge_polyhedron(numpy.array([[0, 1, 1, -3]]))
    actual = input.reducable_columns_approx()
    expected = puan.ndarray.ge_polyhedron(numpy.array([0, 0, -3]))
    assert numpy.array_equal(actual, expected)
    input = puan.ndarray.ge_polyhedron(numpy.array([[2, 1, 1, -1]]))
    actual = input.reducable_columns_approx()
    expected = puan.ndarray.ge_polyhedron(numpy.array([1, 1, -2]))
    assert numpy.array_equal(actual, expected)
    input = puan.ndarray.ge_polyhedron(numpy.array([
        [ 0,-1, 1, 0, 0, 0], # 1
        [ 0, 0,-1, 1, 0, 0], # 2
        [-1,-1, 0,-1, 0, 0], # 3 1+2+3 -> Force not variable 0
    ]))
    actual = input.reducable_columns_approx()
    expected = puan.ndarray.ge_polyhedron(numpy.array([ 0, 0, 0, 0, 0]))
    assert numpy.array_equal(actual, expected)
    input = puan.ndarray.ge_polyhedron(numpy.array([
        [1, 1],
        [1, -1]
    ]))
    actual = input.reducable_columns_approx()
    expected = puan.ndarray.ge_polyhedron(numpy.array([0]))
    assert numpy.array_equal(actual, expected)


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
            pg.Any(*list("xy")),
            pg.Any(*list("ab")),
        ),
        pg.All(*list("mno"))
    )
    assert actual_output == expected_output

def test_parse_short_compound_proposition():

    line_rules = [
        ("Any", ['a', 'b', 'c'], None, None),
        ("AtMost", ['a', 'b', 'c'], 1, None),
        ("Any", ['d', 'e'], None, None),
        ("AtMost", ['d', 'e'], 1, None),
        ("Imply", [("All", ['p', 'a', 'e'], None, None), ("Any", ['x', 'y'], None, None)], None, None),
        ("All", [("All", ['p', 'a', 'e'], None, None), ("AtMost", ['x', 'y'], 1, None)], None, None),
        ("Xor", [("Any", ['p', 'a', 'e'], None, None), ("All", ['x', 'z'], None, None)], None, None),
        ("Imply", [("AtLeast", ['p', 'a', 'e'], 2, None), ("AtMost", ['x', 'z'], 1, None)], None, None),
        ("Xor", [("All", ['a', 'q', 'e'], None, None), ("All", ['f', 'g', 'h'], None, None)], None, None),
    ]
    actual_model = pg.CompoundProposition.from_short(line_rules)
    assert len(actual_model.propositions) == 9

def test_parse_empty_short_compound_proposition():
    actual_model = pg.CompoundProposition.from_short([])
    assert len(actual_model.propositions) == 0

def test_parse_short_compounds_from_text():
    text_lines = [
        # (a & b & (c | d) & (c | e)) -> (a & b & c)
        "('Imply', [('All', ['a','b',('Any',['c','d']),('Any',['c','e'])]),('All', ['a','b','c'])])",
        # (x | d | (z & m)) -xor-> a
        "('Imply', [('Any', ['x','d', ('All', ['z','m'])]), ('Xor',['a'])])"
    ]
    actual_model = pg.CompoundProposition.from_short(list(map(ast.literal_eval, text_lines))) 
    assert len(actual_model.propositions) == 2

def test_parse_short_compund_strings_with_different_combinations():
    line_rule_strs = [
        "('')"
        "(('a','b','c'),('d',)),'REQUIRES_EXCLUSIVELY',('x','y','z'),('x',)",
        "(('a',),),'REQUIRES_ALL',('x',)",
        "(('aa','aa'),),'REQUIRES_ALL',('xx','xx')",
        "(('a')),'REQUIRES_ALL',('b',)",
        "(('a','ab')),'REQUIRES_ALL',('b','ba')",
        "(['xc40','T80']),'REQUIRES_ALL',['o0']",
        "['xxx', ('yyy','zzz')],'REQUIRES_ALL',['aaa','bbb']",
        "['xxx', ['yyy','zzz']],'REQUIRES_ALL',['aaa','bbb']",
        "(['aa', 'bb'], ['cc']),'REQUIRES_ALL',('vv',)",
        "[('aa', 'bb'), ('dd')],'REQUIRES_ALL',('vv',)",
        "[],'REQUIRES_ALL',('x',)",
        "(),'REQUIRES_ALL',('x',)",
    ]
    actual = cc.conjunctional_implication_proposition(
        list(
            map(
                maz.compose(
                    cc.cicR.to_implication_proposition,
                    cc.cicR.from_string
                ),
                line_rule_strs
            )
        )
    )
    assert actual.to_polyhedron().shape == (17,25)

def test_parse_line_rule_when_an_empty_any_condition():

    line_rule_strs = [
        "[],'REQUIRES_ALL',('x'),()",
        "(),'REQUIRES_ALL',('y'),()",
    ]
    actual = list(
        map(
            maz.compose(
                cc.cicR.to_implication_proposition,
                cc.cicR.from_string
            ),
            line_rule_strs
        )
    )
    expected = [
        cc.implication_proposition(
            cc.Operator.ALL,
            cc.conditional_proposition("ALL", [
                cc.boolean_variable_proposition("x")
            ]),
            cc.conditional_proposition("ANY", [])
        ),
        cc.implication_proposition(
            cc.Operator.ALL,
            cc.conditional_proposition("ALL", [
                cc.boolean_variable_proposition("y")
            ])
        ),
    ]
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert a == e

def test_parse_line_rules_when_having_numbers_inside():

    line_rule_strs = [
        "(('$§@£_-/', 'YY-0', 'X_y5')),'REQUIRES_ALL',('1','22','345')"
    ]
    actual = list(
        map(
            maz.compose(
                cc.cicR.to_implication_proposition,
                cc.cicR.from_string
            ),
            line_rule_strs
        )
    )
    expected = [
        cc.implication_proposition(
            cc.Operator.ALL,
            cc.conditional_proposition("ALL", [
                cc.boolean_variable_proposition("1"),
                cc.boolean_variable_proposition("22"),
                cc.boolean_variable_proposition("345"),
            ]),
            cc.conditional_proposition("ALL", [
                cc.boolean_variable_proposition("$§@£_-/"),
                cc.boolean_variable_proposition("YY-0"),
                cc.boolean_variable_proposition("X_y5"),
            ])
        )
    ]
    assert actual[0] == expected[0]

# def test_matrix2cic_lines() -> list:

#     expected_line_rules = [
#         "((),'REQUIRES_ALL',())",
#         "((),'REQUIRES_ALL',('x'))",
#         "((),'REQUIRES_ALL',('x','y'))",
#         "((),'REQUIRES_ALL',('x','y','z'))",
#         "(('a'),'REQUIRES_ALL',())",
#         "(('a'),'REQUIRES_ALL',('x'))",
#         "(('a'),'REQUIRES_ALL',('x','y'))",
#         "(('a'),'REQUIRES_ALL',('x','y','z'))",
#         "(('a','b'),'REQUIRES_ALL',())",
#         "(('a','b'),'REQUIRES_ALL',('x'))",
#         "(('a','b'),'REQUIRES_ALL',('x','y'))",
#         "(('a','b'),'REQUIRES_ALL',('x','y','z'))",
#         "(('a','b','c'),'REQUIRES_ALL',())",
#         "(('a','b','c'),'REQUIRES_ALL',('x'))",
#         "(('a','b','c'),'REQUIRES_ALL',('x','y'))",
#         "(('a','b','c'),'REQUIRES_ALL',('x','y','z'))",
#         "((),'REQUIRES_ANY',())",
#         "((),'REQUIRES_ANY',('x'))",
#         "((),'REQUIRES_ANY',('x','y'))",
#         "((),'REQUIRES_ANY',('x','y','z'))",
#         "(('a'),'REQUIRES_ANY',())",
#         "(('a'),'REQUIRES_ANY',('x'))",
#         "(('a'),'REQUIRES_ANY',('x','y'))",
#         "(('a'),'REQUIRES_ANY',('x','y','z'))",
#         "(('a','b'),'REQUIRES_ANY',())",
#         "(('a','b'),'REQUIRES_ANY',('x'))",
#         "(('a','b'),'REQUIRES_ANY',('x','y'))",
#         "(('a','b'),'REQUIRES_ANY',('x','y','z'))",
#         "(('a','b','c'),'REQUIRES_ANY',())",
#         "(('a','b','c'),'REQUIRES_ANY',('x'))",
#         "(('a','b','c'),'REQUIRES_ANY',('x','y'))",
#         "(('a','b','c'),'REQUIRES_ANY',('x','y','z'))",
#         "((),'FORBIDS_ALL',())",
#         "((),'FORBIDS_ALL',('x'))",
#         "((),'FORBIDS_ALL',('x','y'))",
#         "((),'FORBIDS_ALL',('x','y','z'))",
#         "(('a'),'FORBIDS_ALL',())",
#         "(('a'),'FORBIDS_ALL',('x'))",
#         "(('a'),'FORBIDS_ALL',('x','y'))",
#         "(('a'),'FORBIDS_ALL',('x','y','z'))",
#         "(('a','b'),'FORBIDS_ALL',())",
#         "(('a','b'),'FORBIDS_ALL',('x'))",
#         "(('a','b'),'FORBIDS_ALL',('x','y'))",
#         "(('a','b'),'FORBIDS_ALL',('x','y','z'))",
#         "(('a','b','c'),'FORBIDS_ALL',())",
#         "(('a','b','c'),'FORBIDS_ALL',('x'))",
#         "(('a','b','c'),'FORBIDS_ALL',('x','y'))",
#         "(('a','b','c'),'FORBIDS_ALL',('x','y','z'))",
#         "((),'ONE_OR_NONE',())",
#         "((),'ONE_OR_NONE',('x'))",
#         "((),'ONE_OR_NONE',('x','y'))",
#         "((),'ONE_OR_NONE',('x','y','z'))",
#         "(('a'),'ONE_OR_NONE',())",
#         "(('a'),'ONE_OR_NONE',('x'))",
#         "(('a'),'ONE_OR_NONE',('x','y'))",
#         "(('a'),'ONE_OR_NONE',('x','y','z'))",
#         "(('a','b'),'ONE_OR_NONE',())",
#         "(('a','b'),'ONE_OR_NONE',('x'))",
#         "(('a','b'),'ONE_OR_NONE',('x','y'))",
#         "(('a','b'),'ONE_OR_NONE',('x','y','z'))",
#         "(('a','b','c'),'ONE_OR_NONE',())",
#         "(('a','b','c'),'ONE_OR_NONE',('x'))",
#         "(('a','b','c'),'ONE_OR_NONE',('x','y'))",
#         "(('a','b','c'),'ONE_OR_NONE',('x','y','z'))",
#     ]
#     exp_line_rules = puan.logic.cic.cicEs.from_strings(expected_line_rules).to_cicRs()
#     variables = sorted(exp_line_rules.variables())
#     actual_line_rules = puancore.linalg.matrix.matrix2crc_lines(
#         puan.vmap.value_map2matrix(
#             puancore.logic.crc.line.parsed_linerules2value_map(
#                 puancore.logic.crc.line.index_parsed_line_rules(exp_line_rules, variables, 1)
#             ),
#             len(variables),
#         ),
#         functools.partial(
#             puancore.comp.indexing,
#             variables
#         )
#     )
#     result = list(actual_line_rules)

def test_convert_one_or_none_to_matrix():

    expected_line_rules = [
        "((),'ONE_OR_NONE',())",
        "((),'ONE_OR_NONE',('x'))",
        "((),'ONE_OR_NONE',('x','y'))",
        "((),'ONE_OR_NONE',('x','y','z'))",
        "(('a'),'ONE_OR_NONE',())",
        "(('a'),'ONE_OR_NONE',('x'))",
        "(('a'),'ONE_OR_NONE',('x','y'))",
        "(('a'),'ONE_OR_NONE',('x','y','z'))",
        "(('a','b'),'ONE_OR_NONE',())",
        "(('a','b'),'ONE_OR_NONE',('x'))",
        "(('a','b'),'ONE_OR_NONE',('x','y'))",
        "(('a','b'),'ONE_OR_NONE',('x','y','z'))",
        "(('a','b','c'),'ONE_OR_NONE',())",
        "(('a','b','c'),'ONE_OR_NONE',('x'))",
        "(('a','b','c'),'ONE_OR_NONE',('x','y'))",
        "(('a','b','c'),'ONE_OR_NONE',('x','y','z'))",
    ]

    actual_matrix = cc.conjunctional_implication_proposition(list(
        map(
            maz.compose(
                cc.cicR.to_implication_proposition,
                cc.cicR.from_string
            ),
            expected_line_rules
        )
    )).to_polyhedron()

    expected_matrix_a = numpy.array([
        [-3, -3, -3, -1, -1, -1],
        [-3, -3, -1, -1, -1,  0],
        [-2, -2, -2, -1, -1,  0],
        [-2, -2, -1, -1,  0,  0],
        [-3, -1, -1, -1,  0,  0],
        [-1, -1, -1, -1,  0,  0],
        [-2, -1, -1,  0,  0,  0],
        [-1, -1, -1,  0,  0,  0],
        [-1, -1,  0,  0,  0,  0],
        [-1, -1, -1,  0,  0,  0],
        [-1, -1,  0,  0,  0,  0],
        [-1,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0]],
        dtype=numpy.int16
    )
    expected_vector_b = numpy.array([-10, -7, -7, -5, -4, -4, -3, -3, -2, -1, -1, -1, -1], dtype=numpy.int16)

    sorted_polyhedron = numpy.sort(actual_matrix)
    assert (sorted_polyhedron.A == expected_matrix_a).all()
    assert (sorted_polyhedron.b == expected_vector_b).all()


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

    M = numpy.array([
        [ 1, 0, 1, 0, 0],
        [-3,-2,-1,-1, 0],
    ], dtype="int32")

    rows, cols = puan.ndarray.reducable_rows_and_columns(M)
    assert cols[3] == 0


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
            ])
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
            ])
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
            ])
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
            ])
        ),
    ]
    for inpt, expected_output in test_cases:
        actual_output = inpt.truncate()
        assert (actual_output == expected_output).all()

def test_truncate_documentation_examples():
    """Documentation example"""
    test_cases = [
        (
            puan.ndarray.integer_ndarray([1, 2, 3, 4]),
            puan.ndarray.integer_ndarray([1, 2, 4, 8])),
        (
            puan.ndarray.integer_ndarray([3, 6, 2, 8]),
            puan.ndarray.integer_ndarray([2, 4, 1, 8])),
        (
            puan.ndarray.integer_ndarray([-3, -6, 2, 8]),
            puan.ndarray.integer_ndarray([-2, -4, 1, 8])),
        (
            puan.ndarray.integer_ndarray([1, 1, 2, 2, 2, 3]),
            puan.ndarray.integer_ndarray([ 1,  1,  3,  3,  3, 12])),
        (
            puan.ndarray.integer_ndarray([
                [ 0, 0, 0, 0, 1, 2],
                [-1,-1,-1,-1, 0, 0],
                [ 0, 0, 1, 2, 0, 0],
                [ 0, 0, 1, 0, 0, 0]
                ]),
            puan.ndarray.integer_ndarray([-4, -4, 24, 12,  1,  2])),
        (
            puan.ndarray.integer_ndarray([
                [
                    [ 1,  2,  2,  2,  2],
                    [ 1, -2, -1,  2, -2],
                    [ 1,  0,  2, -1, -1],
                    [ 2,  1,  1,  0,  1]
                ], [
                    [-1,  0, -1,  2,  0],
                    [-1,  1, -2,  2, -1],
                    [ 0, -2,  1, -2,  2],
                    [ 0, -2,  2,  2,  0]
                ], [
                    [ 1, -1,  0,  1,  1],
                    [ 2, -1, -2, -2,  0],
                    [ 2,  2,  1,  1, -2],
                    [ 1, -1,  1,  0,  2]
                ]]),
            puan.ndarray.integer_ndarray([
                [ 8,  2,  2, -1,  2],
                [-1, -4,  4,  4,  2],
                [ 2, -2,  2,  1,  8]])
        )]
    for inpt, expected_output in test_cases:
        actual_output = inpt.truncate()
        assert (actual_output == expected_output).all()

def test_separable():
    """Documentation example"""
    ge = puan.ndarray.ge_polyhedron([0, -2, 1, 1])
    actual_output = ge.separable(numpy.array([
        [1,0,1],
        [1,1,1],
        [0,0,0]]))
    expected_output = numpy.array([True, False, False])
    assert numpy.array_equal(actual_output, expected_output)

def test_ineq_separate_points():
    """Documentation example"""
    input = puan.ndarray.ge_polyhedron(numpy.array([
            [ 0, 1, 0],
            [ 0, 1, -1],
            [ -1, -1, 1]
        ]))
    points = numpy.array([[1, 1], [4, 2]])
    actual = input.ineq_separate_points(points)
    expected = numpy.array([False, False, True])
    assert numpy.array_equal(actual, expected)
    input = puan.ndarray.ge_polyhedron(numpy.array([
                [ 0, 1, 0, -1],
                [ 0, 1, -1, 0],
                [ -1, -1, 1, -1]
            ]))
    points = numpy.array([
            [[1, 1, 1], [4, 2, 1]],
            [[0, 1, 0], [1, 2, 1]]
        ])
    actual = input.ineq_separate_points(points)
    expected = numpy.array([[False, False, True],
                            [False, True, False]])
    assert numpy.array_equal(actual, expected)

def test_or_get():
    """Documentation example"""
    input = dict((("a", 1), ("b", 2)))
    keys = ["1", "a"]
    actual = puan.misc.or_get(input, keys)
    expected = 1
    assert actual == expected
    input = dict((("a", 1), ("b", 2)))
    keys = ["b", "a"]
    actual = puan.misc.or_get(input, keys)
    expected = 2
    assert actual == expected
    input = dict((("a", 1), ("b", 2)))
    keys = [1]
    default_value = 0
    actual = puan.misc.or_get(input, keys, default_value)
    expected = 0
    assert actual == expected

def test_or_replace():
    """Documentation example"""
    input = dict((("a", 1), ("b", 2)))
    keys = ["1", "a"]
    value = "1"
    actual = puan.misc.or_replace(input, keys, value)
    expected = {'a': '1', 'b': 2}
    assert actual == expected
    input = dict((("a", 1), ("b", 2)))
    keys = ["b", "a"]
    value = "1"
    actual = puan.misc.or_replace(input, keys, value)
    expected = {'a': 1, 'b': '1'}
    assert actual == expected

def test_implication_proposition_to_constraints_with_default_value():

    icvp = cc.implication_conjunctional_variable_proposition(
        cc.Operator.ANY, 
        cc.conjunctional_variable_proposition(
            [
                cc.boolean_variable_proposition("x"),
                cc.boolean_variable_proposition("y"),
                cc.boolean_variable_proposition("z"),
            ],
        ),
        cc.conjunctional_variable_proposition(
            [
                cc.boolean_variable_proposition("a"),
                cc.boolean_variable_proposition("b"),
                cc.boolean_variable_proposition("c"),
            ]
        ),
        default=[
            cc.boolean_variable_proposition("z")
        ]
    )

    assert icvp.supporting_variable.id == "abcxy"
    assert sorted(map(operator.itemgetter(1), icvp.to_constraints())) == [
        [-3,-3,-3, 1, 1, 1,-8],
        [-2,-2,-2,-1,-1, 1,-6],
        [ 2, 2, 2, 1, 1,-7, 0],
    ]

def test_conjunction_proposition_to_constraints_with_default_value():

    conj_prop = cc.conjunctional_implication_proposition([
        cc.implication_proposition(
            cc.Operator.XOR, 
            cc.conditional_proposition("ALL", [
                cc.boolean_variable_proposition("x"),
                cc.boolean_variable_proposition("y"),
                cc.boolean_variable_proposition("z"),
            ]),
            cc.conditional_proposition("ALL", [
                cc.boolean_variable_proposition("a"),
                cc.boolean_variable_proposition("b"),
            ]),
            default=[
                cc.boolean_variable_proposition("z")
            ]
        )  
    ])
    assert conj_prop.to_polyhedron().shape == (4,7)
