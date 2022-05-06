import puan
import puan.logic
import numpy
import operator

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
                "ruleType": "RT",
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
                "ruleType": "RT",
                "components": [
                    {"code": "x"},
                    {"code": "y"}
                ]
            }
        },
        {
            "id": "",
            "consequence": {
                "ruleType": "RT",
                "components": [
                    {"code": "z"}
                ]
            }
        },
    ]

    variables = puan.logic.cic.cicJEs.variables(rules, id_ident="code")
    assert set(variables) == set(["a", "b", "c", "x", "y", "z"])

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
                puan.to_value_map(rand_matrix),
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
            "condition": {
                "relation": "ALL",
                "subConditions": [{
                    "relation": "ANY",
                    "components": [
                        {"id": "x"},
                        {"id": "y"}
                    ]
                }]
            },
            "consequence": {
                "ruleType": "REQUIRES_ALL",
                "components": [
                    {"id": "a"},
                    {"id": "b"}
                ]
            }
        },
        {
            "condition": {
                "relation": "ANY",
                "subConditions": [
                    {
                        "relation": "ALL",
                        "components": [
                            {"id": "a"},
                            {"id": "x"}
                        ]
                    },
                    {
                        "relation": "ALL",
                        "components": [
                            {"id":"b"}
                        ]
                    }
                ]
            },
            "consequence": {
                "ruleType": "REQUIRES_EXCLUSIVELY",
                "components": [
                    {"id": "c"},
                    {"id": "d"},
                ]
            }
        }
    ]
    variables = sorted(puan.logic.cic.cicJEs.variables(rules))
    cicrs = puan.logic.cic.cicJEs(rules).to_cicRs()
    matrix = cicrs.to_ge_polyhedron(variables.index)
    expected_feasible_configurations = numpy.array([
       #"a  b  c  d  x  y"
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 0],
        [1, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 0],
        [1, 1, 0, 1, 0, 1],
    ])
    expected_infeasible_configurations = numpy.array([
       #"a  b  c  d  x  y"
        [1, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 0, 1],
    ])

    A, b = matrix.to_linalg()
    assert (numpy.matmul(expected_feasible_configurations, A.T) >= b.T).all(axis=1).all()
    assert (numpy.matmul(expected_infeasible_configurations, A.T) < b.T).any(axis=1).all()

def test_compress_rules():
    rules = [
        {'condition': {
            'relation': "ALL",
            'subConditions': [{
                'relation': "ALL",
                'components': [
                    {'id': 'a'},
                    {'id': 'b'}
                    ]
                }
            ]
        },
        'consequence': {
            'ruleType': 'REQUIRES_ALL',
            'components': [
                {'id': 'x'}
                ]
            }
        },
        {'condition': {
            'relation': "ALL",
            'subConditions': [{
                'relation': "ALL",
                'components': [
                    {'id': 'b'},
                    {'id': 'a'}
                    ]
                }]
            },
        'consequence': {
            'ruleType': 'REQUIRES_ALL',
            'components': [
                {'id': 'x'},
                {'id': 'y'}
                ]
            }
        },
        {'condition': {
            'relation': "ALL",
            'subConditions': [{
                'relation': "ALL",
                'components': [
                    {'id': 'a'},
                    {'id': 'b'}
                    ]
                }]
            },
        'consequence': {
            'ruleType': 'FORBIDS_ALL',
            'components': [
                {'id': 'x'},
                {'id': 'y'}
                ]
            }
        },
        {'condition': {
            'relation': "ALL",
            'subConditions': [{
                'relation': "ALL",
                'components': [
                    {'id': 'a'},
                    {'id': 'b'}
                    ]
                }]
            },
        'consequence': {
            'ruleType': 'FORBIDS_ALL',
            'components': [
                {'id': 'y'},
                {'id': 'z'}
                ]
            }
        },
        {'condition': {
            'relation': "ALL",
            'subConditions': [{
                'relation': "ALL",
                'components': [
                    {'id': 'a'},
                    {'id': 'b'}
                    ]
                }]
            },
        'consequence': {
            'ruleType': 'PREFERRED',
            'components': [
                {'id': 'x'},
                {'id': 'y'}
                ]
            }
        },
        {'condition': {
            'relation': "ALL",
            'subConditions': [{
                'relation': "ALL",
                'components': [
                    {'id': 'a'},
                    {'id': 'b'}
                    ]
                }
            ]
        },
        'consequence': {
            'ruleType': 'PREFERRED',
            'components': [
                {'id': 'y'},
                {'id': 'z'}
                ]
            }
        },
        {'condition': {
            'relation': "ALL",
            'subConditions': [{
                'relation': "ALL",
                'components': [
                    {'id': 'a'},
                    {'id': 'b'}
                    ]
                }
            ]
        },
        'consequence': {
            'ruleType': 'ONE_OR_NONE',
            'components': [
                {'id': 'x'},
                {'id': 'y'}
                ]
            }
        },
        {'condition': {
            'relation': "ALL",
            'subConditions': [{
                'relation': "ALL",
                'components': [
                    {'id': 'a'},
                    {'id': 'b'}
                    ]
                }
            ]
        },
        'consequence': {
            'ruleType': 'ONE_OR_NONE',
            'components': [
                {'id': 'y'},
                {'id': 'z'}
                ]
            }
        },
        {'condition': {
            'relation': "ALL",
            'subConditions': [{
                'relation': "ALL",
                'components': [
                    {'id': 'a'},
                    {'id': 'b'}
                    ]
                }
            ]
        },
        'consequence': {
            'ruleType': 'REQUIRES_EXCLUSIVELY',
            'components': [
                {'id': 'x'},
                {'id': 'y'}
                ]
            }
        },
        {'condition': {
            'relation': "ALL",
            'subConditions': [{
                'relation': "ALL",
                'components': [
                    {'id': 'a'},
                    {'id': 'b'}
                    ]
                }
            ]
        },
        'consequence': {
            'ruleType': 'REQUIRES_EXCLUSIVELY',
            'components': [
                {'id': 'z'}
                ]
            }
        },
        {'condition': {
            'relation': "ALL",
            'subConditions': [{
                'relation': "ALL",
                'components': [
                    {'id': 'a'},
                    {'id': 'b'}
                    ]
                }
            ]
        },
        'consequence': {
            'ruleType': 'REQUIRES_ANY',
            'components': [
                {'id': 'i'}
                ]
            }
        },
        {'condition': {
            'relation': "ALL",
            'subConditions': [{
                'relation': "ALL",
                'components': [
                    {'id': 'a'},
                    {'id': 'b'}
                    ]
                }
            ]
        },
        'consequence': {
            'ruleType': 'REQUIRES_ANY',
            'components': [
                {'id': 'j'},
                {'id': 'k'}
                ]
            }
        }
    ]

    expected_result = [{
        'condition': {
        'relation': "ALL",
        'subConditions': [{
            'relation': "ALL",
            'components': [
                {'id': 'a'},
                {'id': 'b'}
                ]
            }]
        },
        'consequence': {
            'ruleType': 'REQUIRES_ALL',
            'components': [
                {'id': 'i'},  # From require any
                {'id': 'x'},
                {'id': 'y'},
                {'id': 'z'}  # From require exclusively
                ]
            }
        },
        {'condition': {
            'relation': "ALL",
            'subConditions': [{
                'relation': "ALL",
                'components': [
                    {'id': 'a'},
                    {'id': 'b'}
                    ]
                }]
            },
        'consequence': {
            'ruleType': 'FORBIDS_ALL',
            'components': [
                {'id': 'x'},
                {'id': 'y'},
                {'id': 'z'},
                ]
            }
        },
        {'condition': {
            'relation': "ALL",
            'subConditions': [{
                'relation': "ALL",
                'components': [
                    {'id': 'a'},
                    {'id': 'b'}
                    ]
                }]
            },
        'consequence': {
            'ruleType': 'PREFERRED',
            'components': [
                {'id': 'x'},
                {'id': 'y'},
                {'id': 'z'}
                ]
            }
        },
        {'condition': {
            'relation': "ALL",
            'subConditions': [{
                'relation': "ALL",
                'components': [
                    {'id': 'a'},
                    {'id': 'b'}
                    ]
                }]
            },
        'consequence': {
            'ruleType': 'ONE_OR_NONE',
            'components': [
                {'id': 'x'},
                {'id': 'y'}
                ]
            }
        },
        {'condition': {
            'relation': "ALL",
            'subConditions': [{
                'relation': "ALL",
                'components': [
                    {'id': 'a'},
                    {'id': 'b'}
                    ]
                }]
            },
        'consequence': {
            'ruleType': 'ONE_OR_NONE',
            'components': [
                {'id': 'y'},
                {'id': 'z'}
                ]
            }
        },
        {'condition': {
            'relation': "ALL",
            'subConditions': [{
                'relation': "ALL",
                'components': [
                    {'id': 'a'},
                    {'id': 'b'}
                    ]
                }]
            },
        'consequence': {
            'ruleType': 'REQUIRES_EXCLUSIVELY',
            'components': [
                {'id': 'x'},
                {'id': 'y'}
                ]
            }
        },
        {'condition': {
            'relation': "ALL",
            'subConditions': [{
                'relation': "ALL",
                'components': [
                    {'id': 'a'},
                    {'id': 'b'}
                    ]
                }]
            },
        'consequence': {
            'ruleType': 'REQUIRES_ANY',
            'components': [
                {'id': 'j'},
                {'id': 'k'}
                ]
            }
        }
    ]
    actual = puan.logic.cic.cicJEs.compress(rules, id_ident="id")
    for rule in actual:
        rule['consequence']['components'] = sorted(rule['consequence']['components'], key=lambda d: d['id'])
    assert [i for i in actual if i not in expected_result] == []
    assert [i for i in expected_result if i not in actual] == []

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
    rules_gen = puan.logic.sta.application.to_cicJEs(
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
    rules_list = list(rules_gen)
    assert len(rules_list) == 2
    assert len(rules_list[0]["condition"]["subConditions"]) == 1
    assert rules_list[0]["condition"]["subConditions"][0]["relation"] == "ANY"
    assert len(rules_list[0]["condition"]["subConditions"][0]["components"]) == 3
    assert len(rules_list[0]["consequence"]["components"]) == 4

    assert len(rules_list[1]["condition"]["subConditions"]) == 1
    assert rules_list[1]["condition"]["subConditions"][0]["relation"] == "ANY"
    assert len(rules_list[1]["condition"]["subConditions"][0]["components"]) == 2
    assert len(rules_list[1]["consequence"]["components"]) == 4

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
    reducable_rows = puan.reducable_rows(matrix)
    actual = puan.reduce(matrix, rows_vector=reducable_rows)
    expected = numpy.array([
        [-1,-1,-1, 1, 0, 2], # stay
        [ 0,-1, 1, 1, 0, 0], # stay
        [-3,-2,-1,-1, 0, 0], # stay
        [-4,-1,-1,-1,-1,-1], # stay
        [ 2, 1, 1, 1, 1, 1], # stay
        [ 0, 1, 1, 0, 1,-1], # stay
    ])
    assert numpy.array_equal(actual,expected)

def test_reduce():
    """Documentation example"""
    input = puan.ge_polyhedron(numpy.array([
        [ 0,-1, 1, 0, 0, 0, 0],
        [ 0, 0,-1, 1, 0, 0, 0],
        [-1, 0, 0,-1,-1, 0, 0],
        [ 1, 0, 0, 0, 0, 1, 1],
    ]))

    columns_vector = numpy.array([1,0,0,0,0,0])

    actual = input.reduce(columns_vector=columns_vector)
    expected = puan.ge_polyhedron(numpy.array([
            [ 1, 1, 0, 0, 0, 0],
            [ 0,-1, 1, 0, 0, 0],
            [-1, 0,-1,-1, 0, 0],
            [ 1, 0, 0, 0, 1, 1],
        ]))
    assert numpy.array_equal(actual,expected)


def test_reducable_rows_and_columns():
    """Documentation example"""
    matrix = numpy.array([
        [ 0,-1, 1, 0, 0, 0], # 1
        [ 0, 0,-1, 1, 0, 0], # 2
        [-1,-1, 0,-1, 0, 0], # 3 1+2+3 -> Force not variable 0
        [ 1, 0, 0, 0, 1, 0], # Force variable 3
        [ 0, 0, 0, 0, 0,-1], # Force not variable 4
        [ 0, 1, 1, 0, 1, 0], # Redundant rule
        [ 0, 1, 1, 0, 1,-1], # Redundant when variable 4 forced not
    ])

    expected_red_cols_approx = numpy.array([0, 0, 0, 1, -2])
    expected_red_cols = numpy.array([-1, 0, 0, 1, -2])
    expected_red_rows = numpy.array([0, 0, 0, 1, 1, 1, 1])

    actual_red_rows, actual_red_cols = puan.reducable_rows_and_columns(matrix)
    assert numpy.array_equal(actual_red_rows, expected_red_rows)
    assert numpy.array_equal(actual_red_cols, expected_red_cols_approx)

def test_reducable_columns_approx():
    """Documentation examples"""
    input = puan.ge_polyhedron(numpy.array([[0, -1, -1, -1]]))
    actual = input.reducable_columns_approx()
    expected = puan.ge_polyhedron(numpy.array([-2, -2, -2]))
    assert numpy.array_equal(actual, expected)
    input = puan.ge_polyhedron(numpy.array([[3, 1, 1, 1]]))
    actual = input.reducable_columns_approx()
    expected = puan.ge_polyhedron(numpy.array([1, 1, 1]))
    assert numpy.array_equal(actual, expected)
    input = puan.ge_polyhedron(numpy.array([[0, 1, 1, -3]]))
    actual = input.reducable_columns_approx()
    expected = puan.ge_polyhedron(numpy.array([0, 0, -3]))
    assert numpy.array_equal(actual, expected)
    input = puan.ge_polyhedron(numpy.array([[2, 1, 1, -1]]))
    actual = input.reducable_columns_approx()
    expected = puan.ge_polyhedron(numpy.array([1, 1, -2]))
    assert numpy.array_equal(actual, expected)
    input = puan.ge_polyhedron(numpy.array([
        [ 0,-1, 1, 0, 0, 0], # 1
        [ 0, 0,-1, 1, 0, 0], # 2
        [-1,-1, 0,-1, 0, 0], # 3 1+2+3 -> Force not variable 0
    ]))
    actual = input.reducable_columns_approx()
    expected = puan.ge_polyhedron(numpy.array([ 0, 0, 0, 0, 0]))
    assert numpy.array_equal(actual, expected)
    input = puan.ge_polyhedron(numpy.array([
        [1, 1],
        [1, -1]
    ]))
    actual = input.reducable_columns_approx()
    expected = puan.ge_polyhedron(numpy.array([0]))
    assert numpy.array_equal(actual, expected)

def test_reduce_columns():
    """Documentation example"""
    input = puan.ge_polyhedron(numpy.array([
        [0,-1, 1, 0, 0],
        [0, 0,-1, 1, 0],
        [0, 0, 0,-1, 1],
    ]))

    columns_vector = numpy.array([1, 0, -1, 0]) # meaning assume index 0 and not assume index 2
    actual = input.reduce_columns(columns_vector)
    expected = puan.ge_polyhedron(numpy.array([
                        [1, 1, 0],
                        [0,-1, 0],
                        [0, 0, 1],
                    ]))
    assert numpy.array_equal(actual, expected)

def test_reducable_rows():
    """Documentation example"""
    input = puan.ge_polyhedron(numpy.array([[-3, -1, -1, 1, 0]]))
    actual = input.reducable_rows()
    expected = puan.ge_polyhedron(numpy.array([True]))
    assert numpy.array_equal(actual, expected)
    input = puan.ge_polyhedron(numpy.array([[0, 1, 1, 1, 0]]))
    actual = input.reducable_rows()
    expected = puan.ge_polyhedron(numpy.array([True]))
    assert numpy.array_equal(actual, expected)

def test_reduce_rows():
    """Documentation example"""
    input = puan.ge_polyhedron(numpy.array([
        [0,-1, 1, 0, 0], # Reduce
        [0, 0,-1, 1, 0], # Keep
        [0, 0, 0,-1, 1], # Reduce
    ]))
    rows_vector = numpy.array([1, 0, 1])
    actual = input.reduce_rows(rows_vector)
    expected = puan.ge_polyhedron(numpy.array([
        [0, 0,-1, 1, 0],
    ]))
    assert numpy.array_equal(actual, expected)
    input = puan.ge_polyhedron(numpy.array([
        [0,-1, 1, 0, 0], # Reduce
        [0, 0,-1, 1, 0], # Keep
        [0, 0, 0,-1, 1], # Reduce
    ]))
    rows_vector = numpy.array([True, False, True])
    actual = input.reduce_rows(rows_vector)
    expected = puan.ge_polyhedron(numpy.array([
        [0, 0,-1, 1, 0],
    ]))
    assert numpy.array_equal(actual, expected)



def test_split_ruleset():
    rules = [
        {'condition': {
            'relation': "ALL",
            'subConditions': [{
                'relation': "ALL",
                'components': [
                    {'id': 'a'},
                    {'id': 'b'}
                    ]
                }]
            },
        'consequence': {
            'ruleType': 'REQUIRES_ANY',
            'components': [
                {'id': 'j'},
                {'id': 'k'}
                ]
            }
        },
        {'condition': {
            'relation': "ALL",
            'subConditions': [{
                'relation': "ANY",
                'components': [
                    {'id': 'b'},
                    {'id': 'c'}
                    ]
                }]
            },
        'consequence': {
            'ruleType': 'REQUIRES_ALL',
            'components': [
                {'id': 'l'},
                {'id': 'm'}
                ]
            }
        },
        {'condition': {
            'relation': "ALL",
            'subConditions': []
            },
        'consequence': {
            'ruleType': 'REQUIRES_ALL',
            'components': [
                {'id': 'a'},
                {'id': 'n'}
                ]
            }
        },
        {'condition': {
            'relation': "ALL",
            'subConditions': []
            },
        'consequence': {
            'ruleType': 'REQUIRES_ALL',
            'components': [
                {'id': 'o'},
                {'id': 'p'}
                ]
            }
        },
        {'condition': {
            'relation': "ALL",
            'subConditions': [{
                'relation': "ALL",
                'components': [
                    {'id': 'p'},
                    ]
                }]
            },
        'consequence': {
            'ruleType': 'REQUIRES_ALL',
            'components': [
                {'id': 'q'},
                ]
            }
        }
    ]
    expected_result = [
        [
            {'condition': {'relation': 'ALL', 'subConditions': [{'relation': 'ALL', 'components': [{'id': 'a'}, {'id': 'b'}]}]}, 'consequence': {'ruleType': 'REQUIRES_ANY', 'components': [{'id': 'j'}, {'id': 'k'}]}},
            {'condition': {'relation': 'ALL', 'subConditions': []}, 'consequence': {'ruleType': 'REQUIRES_ALL', 'components': [{'id': 'a'}, {'id': 'n'}]}},
            {'condition': {'relation': 'ALL', 'subConditions': [{'relation': 'ANY', 'components': [{'id': 'b'}, {'id': 'c'}]}]}, 'consequence': {'ruleType': 'REQUIRES_ALL', 'components': [{'id': 'l'}, {'id': 'm'}]}}
        ],
        [
            {'condition': {'relation': 'ALL', 'subConditions': []}, 'consequence': {'ruleType': 'REQUIRES_ALL', 'components': [{'id': 'o'}, {'id': 'p'}]}},
            {'condition': {'relation': 'ALL', 'subConditions': [{'relation': 'ALL', 'components': [{'id': 'p'}]}]}, 'consequence': {'ruleType': 'REQUIRES_ALL', 'components': [{'id': 'q'}]}}
        ]
    ]
    split_ruleset = puan.logic.cic.cicJEs.split(rules)
    assert split_ruleset == expected_result

def test_cicJE_to_cicE():

    json_rule = {
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
    }

    id_ident = "id"

    expected_output = [(['x','y'],['a','b']),'REQUIRES_ALL',{'m','n','o'},()]
    actual_output = puan.logic.cic.cicJE.to_cicE(json_rule, id_ident)
    assert actual_output[0] == expected_output[0]

def test_parsed_linerules2value_map():

    line_rules = [
        (set(), 'REQUIRES_ANY', ('a', 'b', 'c')),
        (set(), 'ONE_OR_NONE', ('a', 'b', 'c')),
        (set(), 'REQUIRES_ANY', ('d', 'e')),
        (set(), 'ONE_OR_NONE', ('d', 'e')),
        ({'p', 'a', 'e'}, 'REQUIRES_ANY', ('x', 'y')),
        ({'p', 'a', 'e'}, 'ONE_OR_NONE', ('x', 'y')),
        ({'p', 'a', 'e'}, 'REQUIRES_ANY', ('x', 'z')),
        ({'p', 'a', 'e'}, 'ONE_OR_NONE', ('x', 'z')),
        ({'b', 'p', 'd'}, 'REQUIRES_ANY', ('x', 'y')),
        ({'b', 'p', 'd'}, 'ONE_OR_NONE', ('x', 'y')),
        ({'b', 'p', 'd'}, 'REQUIRES_ANY', ('x', 'z')),
        ({'b', 'p', 'd'}, 'ONE_OR_NONE', ('x', 'z')),
        ({'p', 'd', 'a'}, 'REQUIRES_ANY', ('x', 'z')),
        ({'p', 'd', 'a'}, 'ONE_OR_NONE', ('x', 'z')),
        ({'p', 'd', 'a'}, 'REQUIRES_ANY', ('x', 'y')),
        ({'p', 'd', 'a'}, 'ONE_OR_NONE', ('x', 'y')),
        ({'c', 'p', 'a'}, 'REQUIRES_ANY', ('x', 'z')),
        ({'c', 'p', 'a'}, 'ONE_OR_NONE', ('x', 'z')),
        ({'c', 'p', 'a'}, 'REQUIRES_ANY', ('x', 'y')),
        ({'c', 'p', 'a'}, 'ONE_OR_NONE', ('x', 'y')),
        ({'a', 'q', 'e'}, 'REQUIRES_ALL', ('f', 'g', 'h'))
    ]

    actual = puan.logic.cic.cicRs(line_rules).to_ge_polyhedron(puan.logic.cic.cicRs(line_rules).variables().index)
    for v in numpy.nditer(actual):
        try:
            int(v)
        except:
            raise Exception("found value not being int in value map")

def test_parse_empty_line_rule_should_yield_no_variables():
    m = ["(()),'PREFERRED',(),()"]
    actual = puan.logic.cic.cicEs.from_strings(m)
    for parsed in actual:
        cond, _, cons, _ = parsed
        assert len(cond) == 0
        assert len(cons) == 0

def test_parte_line_rules_from_text():
    text_lines = [
        "(('a'),('b'),['c','d'],['c','e']), 'REQUIRES_ALL',         ('a','b','c'),()",
        "[['x','d'], ('z','m')],            'REQUIRES_EXCLUSIVELY', ('a',),()",
        "[['x','d'], ('z','m')],            'FORBIDS_ALL',          ('a',),()"
    ]
    actual = puan.logic.cic.cicEs.from_strings(text_lines).to_cicRs()
    expected = [
        ({'a', 'b', 'c'},       'REQUIRES_ALL', {'a', 'b', 'c'}, ()),
        ({'a', 'b', 'e', 'c'},  'REQUIRES_ALL', {'a', 'b', 'c'}, ()),
        ({'a', 'b', 'c', 'd'},  'REQUIRES_ALL', {'a', 'b', 'c'}, ()),
        ({'a', 'b', 'e', 'd'},  'REQUIRES_ALL', {'a', 'b', 'c'}, ()),
        ({'d'},                 'REQUIRES_ANY', {'a'}, ()),
        ({'x'},                 'REQUIRES_ANY', {'a'}, ()),
        ({'m', 'z'},            'REQUIRES_ANY', {'a'}, ()),
        ({'d'},                 'ONE_OR_NONE',  {'a'}, ()),
        ({'x'},                 'ONE_OR_NONE',  {'a'}, ()),
        ({'m', 'z'},            'ONE_OR_NONE',  {'a'}, ()),
        ({'d'},                 'FORBIDS_ALL',  {'a'}, ()),
        ({'x'},                 'FORBIDS_ALL',  {'a'}, ()),
        ({'m', 'z'},            'FORBIDS_ALL',  {'a'}, ()),
    ]
    assert len(actual) == len(expected)
    for a in actual:
        assert a in expected

def test_parse_line_rule_strings_with_different_combinations():
    line_rule_strs = [
        "(('a',),),'REQUIRES_ALL',('x',)",
        "(('aa','aa'),),'REQUIRES_ALL',('xx','xx')",
        "(('a')),'REQUIRES_ALL',('b',)",
        "(('a','ab')),'REQUIRES_ALL',('b','ba')",
        "(('a','b','c'),('d',)),'REQUIRES_EXCLUSIVELY',('x','y','z'),('x',)",
        "(['xc40','T80']),'REQUIRES_ALL',['o0']",
        "['xxx', ('yyy','zzz')],'REQUIRES_ALL',['aaa','bbb']",
        "['xxx', ['yyy','zzz']],'REQUIRES_ALL',['aaa','bbb']",
        "(['aa', 'bb'], ['cc']),'REQUIRES_ALL',('vv',)",
        "[('aa', 'bb'), ('dd')],'REQUIRES_ALL',('vv',)",
        "[],'REQUIRES_ALL',('x',)",
        "(),'REQUIRES_ALL',('x',)",
    ]
    actual = puan.logic.cic.cicEs.from_strings(line_rule_strs).to_cicRs()
    expected = [
        ({'a'}, 'REQUIRES_ALL', {'x'}, ()),
        ({'aa'}, 'REQUIRES_ALL', {'xx'}, ()),
        ({'a'}, 'REQUIRES_ALL', {'b'}, ()),
        ({'a','ab'}, 'REQUIRES_ALL', {'b','ba'}, ()),
        ({'a','b','c','d'}, 'REQUIRES_ANY', {'x','y','z'}, ('x',)),
        ({'a','b','c','d'}, 'ONE_OR_NONE', {'x','y','z'}, ('x',)),
        ({'xc40'}, 'REQUIRES_ALL', {'o0'}, ()),
        ({'T80'}, 'REQUIRES_ALL', {'o0'}, ()),
        ({'xxx'}, 'REQUIRES_ALL', {'aaa','bbb'}, ()),
        ({'yyy','zzz'}, 'REQUIRES_ALL', {'aaa','bbb'}, ()),
        ({'xxx'}, 'REQUIRES_ALL', {'aaa','bbb'}, ()),
        ({'yyy'}, 'REQUIRES_ALL', {'aaa','bbb'}, ()),
        ({'zzz'}, 'REQUIRES_ALL', {'aaa','bbb'}, ()),
        ({'aa','cc'}, 'REQUIRES_ALL', {'vv'}, ()),
        ({'bb','cc'}, 'REQUIRES_ALL', {'vv'}, ()),
        ({'aa','bb'}, 'REQUIRES_ALL', {'vv'}, ()),
        ({'dd'}, 'REQUIRES_ALL', {'vv'}, ()),
        ({}, 'REQUIRES_ALL', {'x'}, ())
    ]
    for a in actual:
        assert a in expected

def test_parse_line_rule_when_an_empty_any_condition():

    line_rule_strs = [
        "[],'REQUIRES_ALL',('x'),()",
        "(),'REQUIRES_ALL',('y'),()",
    ]
    actual = puan.logic.cic.cicEs.from_strings(line_rule_strs).to_cicRs()
    expected = [
        ({}, 'REQUIRES_ALL', {'x'}, ()),
        ({}, 'REQUIRES_ALL', {'y'}, ())
    ]
    assert len(actual) == len(expected)
    for a in actual:
        assert a in expected

def test_parse_line_rules_when_having_numbers_inside():

    line_rule_strs = [
        "(('$§@£_-/', 'YY-0', 'X_y5')),'REQUIRES_ALL',('1','22','345')"
    ]
    actual = puan.logic.cic.cicEs.from_strings(line_rule_strs).to_cicRs()
    expected = [
        ({"$§@£_-/", "YY-0", "X_y5"}, "REQUIRES_ALL", {"1", "22", "345"}, ())
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

    cicrs = puan.logic.cic.cicEs.from_strings(expected_line_rules).to_cicRs()
    variables = sorted(cicrs.variables())
    actual_matrix = cicrs.to_ge_polyhedron(variables.index)

    expected_matrix = numpy.array([
        [ -1,   0,   0,   0,   0,   0,   0],
        [ -1,   0,   0,   0,  -1,   0,   0],
        [ -1,   0,   0,   0,  -1,  -1,   0],
        [ -1,   0,   0,   0,  -1,  -1,  -1],
        [ -1,   0,   0,   0,   0,   0,   0],
        [ -2,  -1,   0,   0,  -1,   0,   0],
        [ -3,  -2,   0,   0,  -1,  -1,   0],
        [ -4,  -3,   0,   0,  -1,  -1,  -1],
        [ -1,   0,   0,   0,   0,   0,   0],
        [ -3,  -1,  -1,   0,  -1,   0,   0],
        [ -5,  -2,  -2,   0,  -1,  -1,   0],
        [ -7,  -3,  -3,   0,  -1,  -1,  -1],
        [ -1,   0,   0,   0,   0,   0,   0],
        [ -4,  -1,  -1,  -1,  -1,   0,   0],
        [ -7,  -2,  -2,  -2,  -1,  -1,   0],
        [-10,  -3,  -3,  -3,  -1,  -1,  -1]
    ])

    assert (actual_matrix == expected_matrix).all()

def test_neglectable_columns():
    """Documentation example"""
    # Case 1: keep common pattern
    inputs = (
        puan.ge_polyhedron(numpy.array([  # M
                    [-1,-1,-1, 0, 0, 0, 1],
                    [-1,-1, 0,-1, 0, 0, 1],
            ])),
        numpy.array([  # patterns
                    [1, 1, 0],
                    [0, 1, 1],
                    [1, 0, 1]
            ])
    )
    actual = puan.neglectable_columns(*inputs)
    expected = numpy.array([0, 1, 1, 0, 0, 0])
    assert numpy.array_equal(actual, expected)

    # Case 2: neglect common pattern
    inputs = (
        numpy.array([  # M
                    [-1,-1,-1, 0, 0, 0, 1],
                    [-1,-1, 0,-1, 0, 0, 1],
            ]),
        numpy.array([  # patterns
                    [1, 1, 0],
                    [1, 0, 1],
                    [1, 0, 0]
            ])
    )
    actual = puan.neglectable_columns(*inputs)
    expected = numpy.array([1, 0, 0, 0, 0, 0])
    assert numpy.array_equal(actual, expected)

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
    actual = puan.neglect_columns(*inputs)
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
    actual = puan.boolean_ndarray.from_list(*inputs).to_value_map()
    expected = {
        1: [
            [0, 0, 0, 1, 1, 1],
            [0, 1, 2, 2, 3, 4]
        ]
    }
    assert actual == expected

def test_polyhedron2value_map():
    """Documentation example"""
    inputs = numpy.array([
                 [0,-1, 1, 0, 0],
                 [0, 0,-1, 1, 0],
                 [0, 0, 0,-1, 1],
             ])
    actual = puan.ge_polyhedron(inputs).to_value_map()
    expected = {
        1: [[0, 1, 2], [2, 3, 4]],
        -1: [[0, 1, 2], [1, 2, 3]]
        }
    assert actual == expected

def test_polyhedron2linalg():
    """Documentation example"""
    inputs = numpy.array([
                 [0,-1, 1, 0, 0],
                 [0, 0,-1, 1, 0],
                 [0, 0, 0,-1, 1],
             ])
    actual = puan.ge_polyhedron(inputs).to_linalg()
    expected = (numpy.array([
                    [-1, 1, 0, 0],
                    [0, -1, 1, 0],
                    [0, 0, -1, 1]]),
                numpy.array([0,0,0]))
    assert numpy.array_equal(actual[0], expected[0])
    assert numpy.array_equal(actual[1], expected[1])

def test_reducable_matrix_columns_should_keep_zero_columns():

    """
        When reducing columns, we should not say anything
        about "zero"-columns.
    """

    M = numpy.array([
        [ 1, 0, 1, 0, 0],
        [-3,-2,-1,-1, 0],
    ], dtype="int32")

    rows, cols = puan.reducable_rows_and_columns(M)
    assert cols[3] == 0


def test_truncate():
    test_cases = [
        (
            puan.integer_ndarray([
                [
                    [-4, 1, 2,-4,-4,-4],
                    [ 0, 0, 0, 1, 0, 0],
                ],
            ]),
            puan.integer_ndarray([
                [-4, 1, 2, 16,-4,-4]
            ])
        ),
        (
            puan.integer_ndarray([
                [
                    [ 0, 0, 0, 0, 0, 0],
                    [ 0, 0, 0, 0, 0, 0],
                    [ 0, 0, 0, 0, 0, 0],
                    [ 1, 2, 3, 4, 5, 6]
                ],
            ]),
            puan.integer_ndarray([
                [ 1, 2, 4, 8, 16, 32],
            ])
        ),
        (
            puan.integer_ndarray([
                [ 0, 0, 1, 0, 2, 3],
                [-1,-1,-1,-1, 0, 0],
                [ 0, 0, 1, 2, 0, 0],
                [ 0, 0, 1, 0, 0, 0]
            ]),
            puan.integer_ndarray([
                -4, -4, 24, 12, 1, 2
            ])
        ),
        (
            puan.integer_ndarray([
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
            puan.integer_ndarray([
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
            puan.integer_ndarray([1, 2, 3, 4]),
            puan.integer_ndarray([1, 2, 4, 8])),
        (
            puan.integer_ndarray([3, 6, 2, 8]),
            puan.integer_ndarray([2, 4, 1, 8])),
        (
            puan.integer_ndarray([-3, -6, 2, 8]),
            puan.integer_ndarray([-2, -4, 1, 8])),
        (
            puan.integer_ndarray([1, 1, 2, 2, 2, 3]),
            puan.integer_ndarray([ 1,  1,  3,  3,  3, 12])),
        (
            puan.integer_ndarray([
                [ 0, 0, 0, 0, 1, 2],
                [-1,-1,-1,-1, 0, 0],
                [ 0, 0, 1, 2, 0, 0],
                [ 0, 0, 1, 0, 0, 0]
                ]),
            puan.integer_ndarray([-4, -4, 24, 12,  1,  2])),
        (
            puan.integer_ndarray([
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
            puan.integer_ndarray([
                [ 8,  2,  2, -1,  2],
                [-1, -4,  4,  4,  2],
                [ 2, -2,  2,  1,  8]])
        )]
    for inpt, expected_output in test_cases:
        actual_output = inpt.truncate()
        assert (actual_output == expected_output).all()

def test_separable():
    """Documentation example"""
    ge = puan.ge_polyhedron([0, -2, 1, 1])
    actual_output = ge.separable(numpy.array([
        [1,0,1],
        [1,1,1],
        [0,0,0]]))
    expected_output = numpy.array([True, False, False])
    assert numpy.array_equal(actual_output, expected_output)

def test_ineq_separate_points():
    """Documentation example"""
    input = puan.ge_polyhedron(numpy.array([
            [ 0, 1, 0],
            [ 0, 1, -1],
            [ -1, -1, 1]
        ]))
    points = numpy.array([[1, 1], [4, 2]])
    actual = input.ineq_separate_points(points)
    expected = numpy.array([False, False, True])
    assert numpy.array_equal(actual, expected)
    input = puan.ge_polyhedron(numpy.array([
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
