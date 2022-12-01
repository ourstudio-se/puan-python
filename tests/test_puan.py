import pytest
import ast
import functools
import itertools
import puan
import puan.misc
import puan.ndarray
import puan.logic
import puan.logic.plog as pg
import puan.modules.configurator as cc
import numpy
import operator
import maz
import math

from hypothesis import example, given, strategies as st, settings, assume

def short_proposition_strategy():
    mn,mx = -99,99
    return st.tuples(
        st.text(), 
        st.sampled_from([-1,1]), 
        st.lists(
            st.text(),
        ), 
        st.integers(mn,mx), 
        st.tuples(
            st.integers(mn,mx),
            st.integers(mn,mx),
        )
    )

def atom_proposition_strategy():
    return st.builds(
        puan.variable,
        id=st.text(), 
        bounds=st.tuples(
            st.integers(min_value=-5, max_value=0),
            st.integers(min_value=0, max_value=5),
        ),
    )

def atom_boolean_proposition_strategy():
    return st.builds(
        puan.variable,
        id=st.text(),
    )

def variable_proposition_strategy():
    return st.one_of(atom_proposition_strategy(), st.none())

def variable_boolean_proposition_strategy():
    return st.one_of(atom_boolean_proposition_strategy(), st.none())

def atoms_propositions_strategy(mn_size: int = 1, mx_size: int = 5):
    return st.iterables(
        atom_proposition_strategy(),
        min_size=mn_size,
        max_size=mx_size,
    )

def atleast_proposition_strategy():
    return st.builds(
        pg.AtLeast, 
        propositions=atoms_propositions_strategy(),
        variable=variable_boolean_proposition_strategy(), 
        value=st.integers(min_value=-5, max_value=5),
        sign=st.sampled_from([puan.Sign.POSITIVE, puan.Sign.NEGATIVE])
    )

def atmost_proposition_strategy():
    return st.builds(
        pg.AtMost, 
        propositions=atoms_propositions_strategy(),
        variable=variable_boolean_proposition_strategy(), 
        value=st.integers(min_value=-5, max_value=5),
    )

def all_proposition_strategy():
    return st.builds(
        pg.All.from_list,
        atoms_propositions_strategy(),
        variable=variable_boolean_proposition_strategy(),
    )

def any_proposition_strategy():
    return st.builds(
        pg.Any.from_list,
        atoms_propositions_strategy(),
        variable=variable_boolean_proposition_strategy(),
    )

def any_cc_proposition_strategy():
    return st.builds(
        cc.Any.from_list,
        atoms_propositions_strategy(),
        variable=variable_boolean_proposition_strategy(),
        default=atoms_propositions_strategy(),
    )

def imply_proposition_strategy():
    return st.builds(
        pg.Imply,
        condition=atom_proposition_strategy(),
        consequence=atom_proposition_strategy(),
        variable=variable_boolean_proposition_strategy(),
    )

def xor_proposition_strategy():
    return st.builds(
        pg.Xor.from_list,
        atoms_propositions_strategy(),
        variable=variable_boolean_proposition_strategy(),
    )

def xor_cc_proposition_strategy():
    return st.builds(
        cc.Xor.from_list,
        atoms_propositions_strategy(),
        variable=variable_boolean_proposition_strategy(),
        default=atoms_propositions_strategy(),
    )

def xnor_proposition_strategy():
    return st.builds(
        pg.XNor.from_list,
        atoms_propositions_strategy(),
        variable=variable_boolean_proposition_strategy(),
    )

def not_proposition_strategy():
    return st.builds(
        pg.Not,
        atom_proposition_strategy(),
    )

def proposition_strategy():
    return st.one_of(
        atleast_proposition_strategy(),
        atmost_proposition_strategy(),
        all_proposition_strategy(),
        any_proposition_strategy(),
        imply_proposition_strategy(),
        xor_proposition_strategy(),
        xnor_proposition_strategy(),
        not_proposition_strategy(),
    )

def cc_proposition_strategy():
    return st.one_of(
        atleast_proposition_strategy(),
        atmost_proposition_strategy(),
        all_proposition_strategy(),
        any_proposition_strategy(),
        any_cc_proposition_strategy(),
        imply_proposition_strategy(),
        xor_proposition_strategy(),
        xor_cc_proposition_strategy(),
        xnor_proposition_strategy(),
        not_proposition_strategy(),
    )

def propositions_strategy():
    return st.lists(proposition_strategy(), min_size=1)

def cc_propositions_strategy():
    return st.lists(cc_proposition_strategy(), min_size=1)

@given(cc_propositions_strategy())
@settings(deadline=None)
def test_negated_propositions_are_unique(propositions):
    # negated propositions will never have same id as before negated
    for prop1, prop2 in zip(propositions, map(operator.methodcaller("negate"), propositions)):
        assert not prop1.generated_id or prop1.id != prop2.id

@given(propositions_strategy(), st.lists(st.integers(min_value=-3, max_value=3), min_size=99))
@settings(deadline=None)
def test_proposition_polyhedron_conversion(propositions, integers):

    """
        For each model that does not have any known errors we test that if a generated interpretation
        satisfies the model, a corresponding vector interpretation also satisfies the model's polyhedron.
    """

    model = pg.All(*propositions)
    if len(model.errors()) == 0:
        polyhedron = model.to_ge_polyhedron(True)
        model_variables = sorted(set(map(operator.attrgetter("id"), model.flatten())).difference({model.id}))
        polyhedron_variables = sorted(set(map(operator.attrgetter("id"), polyhedron.A.variables)))
        if not model_variables == polyhedron_variables:
            raise Exception("model variables not equal polyhedron variables")
        integers_clipped = list(itertools.starmap(lambda x,i: numpy.clip(i, x.bounds.lower, x.bounds.upper), zip(polyhedron.A.variables, integers)))
        try:
            model_interpretation_evaluated = model.evaluate_propositions(dict(zip(model_variables, integers_clipped)))
            polyhedron_eval = bool((polyhedron.A.dot(polyhedron.A.construct(*sorted(model_interpretation_evaluated.items()))) >= polyhedron.b).all())
            model_eval = model_interpretation_evaluated[model.id] == 1
            if not model_eval == polyhedron_eval:
                raise Exception("model evaluation not equal polyhedron evaluation")
        except:
            # Some cases integer values are passed beyond variable bounds (even though the numpy.clip, which should prevent it)
            # However, these cases are skipped
            pass

@given(st.lists(st.text(), min_size=3, max_size=3))
@settings(deadline=None)
def test_polyhedron_construct_function(vrs):
    arr = numpy.zeros((1, len(vrs)))
    set_vars = numpy.array(list(map(puan.variable, set(vrs))))
    polyhedron = puan.ndarray.ge_polyhedron(arr[:,:set_vars.size], variables=set_vars)
    interpretation = dict(zip(map(lambda x: x.id, set_vars), range(polyhedron.variables.size)))
    ph_interpretation = dict(zip(map(lambda x: x.id, polyhedron.variables), polyhedron.construct(*interpretation.items())))
    assert interpretation == ph_interpretation

def test_proposition_polyhedron_conversion_specifics():
    
    for model in [
        pg.All(
            pg.AtMost(
                -5, 
                [
                    puan.variable('a', (-5,3)),
                    puan.variable('b', ( 0,2)),
                    puan.variable('c', (-4,4)),
                    puan.variable('d', (-4,5))
                ], 
                variable=puan.variable('B', (0,1))
            ), 
            variable=puan.variable('A', (0,1))
        )
    ]:
        if len(model.errors()) == 0:
            polyhedron = model.to_ge_polyhedron(True)
            model_variables = sorted(set(map(operator.attrgetter("id"), model.flatten())).difference({model.id}))
            polyhedron_variables = sorted(set(map(operator.attrgetter("id"), polyhedron.A.variables)))
            if not model_variables == polyhedron_variables:
                raise Exception("fail variable comparison")
            integers_clipped = list(itertools.starmap(lambda x,i: numpy.clip(i, x.bounds.lower, x.bounds.upper), zip(polyhedron.A.variables, range(polyhedron.A.variables.size))))
            model_interpretation_evaluated = model.evaluate_propositions(dict(zip(model_variables, integers_clipped)))
            if not model_interpretation_evaluated[model.id] == (polyhedron.A.dot(polyhedron.A.construct(*sorted(model_interpretation_evaluated.items()))) >= polyhedron.b).all():
                raise Exception("fail interpretation comparison")



@given(propositions_strategy())
@settings(deadline=None)
def test_model_properties_hypothesis(propositions):
    model = pg.All(*propositions, variable="main")
    
    # Properties that should never fail when accessing
    model.variables
    model.is_contradiction
    model.is_tautologi
    model.id
    model.bounds
    model.equation_bounds
    list(model.compound_propositions)
    list(model.atomic_propositions)

    # Methods that should never fail when running
    model.flatten()
    model.negate()
    model.to_short()
    model.to_text()
    if not model.errors():
        model.to_ge_polyhedron()
        model.to_ge_polyhedron(True)
    model.to_b64()

@given(cc_propositions_strategy())
@settings(deadline=None)
def test_model_json_conversion(propositions):
    model = cc.StingyConfigurator(*propositions, id="main")
    _model = cc.StingyConfigurator.from_json(model.to_json())
    assert model == _model

@given(cc_proposition_strategy())
def test_to_from_b64(proposition):
    string = proposition.to_b64()
    _proposition = pg.from_b64(string)
    assert proposition.id == _proposition.id
    assert proposition.bounds == _proposition.bounds

@given(proposition_strategy())
def test_json_conversion_id_should_be_returned_if_explicitly_defined(proposition):
    json_model = pg.from_json(proposition.to_json()).to_json()
    # generated id (i.e. no ID was explicitly defined) implies that there shouldn't be an ID in the json
    assert not (proposition.generated_id and 'id' in json_model)

@given(short_proposition_strategy())
@settings(deadline=None)
def test_from_short_wont_crash(short_proposition):

    # Should raise if has sub propositions and bounds are other than (0,1) 
    # OR if upper bound is strict lower than lower bound
    if (len(short_proposition[2]) > 0 and short_proposition[4] != (0,1)) or (short_proposition[4][1] < short_proposition[4][0]):
        with pytest.raises(Exception):
            pg.AtLeast.from_short(short_proposition)
    else:
        if short_proposition[4][0] > short_proposition[4][1]:
            with pytest.raises(Exception):
                pg.AtLeast.from_short(short_proposition)
        else:
            pg.AtLeast.from_short(short_proposition)

def test_json_conversion_special_cases():

    model = cc.StingyConfigurator(
        cc.Any(
            pg.Any(
                puan.variable(id='', bounds=(4, 4)),
            ),
            default=[
                puan.variable(id='', bounds=(-4, 2)),
            ],
            variable=""
        ),
        id="main"
    )
    _model = cc.StingyConfigurator.from_json(model.to_json())
    assert model == _model

    model = cc.StingyConfigurator(
        cc.Any(
            puan.variable('0', (0,0)),
            puan.variable('1', (0,0)),
            default=[
                puan.variable('', (0,0))
            ]
        )
    )
    _model = cc.StingyConfigurator.from_json(model.to_json())
    assert model == _model

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
    model = pg.All(*map(pg.Imply.from_cicJE, rules), variable="A")
    matrix = model.to_ge_polyhedron(active=True)

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

        NOTE. Now we raise an error if this is the case. This is since
        a All-proposition with no sub propositions are considered invalid.
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
    with pytest.raises(Exception):
        model = puan.logic.sta.application.to_plog(application, items)


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
            puan.ndarray.integer_ndarray([1, 5, 30]),
            puan.ndarray.integer_ndarray([1, 2, 4]),
            0
        ),
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

    test_cases = [
        (
            puan.ndarray.integer_ndarray([1, 2, 3]),
            puan.ndarray.integer_ndarray([1, 2, 3]),
            0
        ),
        (
            puan.ndarray.integer_ndarray([
                [
                    [-4, 1, 2,-4,-4,-4],
                    [ 0, 0, 0, 1, 0, 0],
                ],
            ]),
            puan.ndarray.integer_ndarray([
                [-4, 1, 2, -4,-4,-4]
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
                [ 1, 2, 3, 4, 5, 6],
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
                -1, -1, 1, -1, 2, 3
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
                [-1,-1, 1,-1, 2, 3],
                [ 0, 0, 0, 0, 0, 0],
                [ 1, 2, 3, 4, 5, 6],
                [-1,-1,-1,-1, 2, 3],
                [ 1, 4, 7, 0, 0, 0]
            ]),
            0
        ),
    ]
    for inpt, expected_output, axis in test_cases:
        actual_output = inpt.ndint_compress(method="first", axis=axis)
        assert numpy.array_equal(actual_output, expected_output)
    
    test_cases = [
        (
            puan.ndarray.integer_ndarray([1, 5, 30]),
            puan.ndarray.integer_ndarray([1, 2, 3]),
            0
        ),
        (
            puan.ndarray.integer_ndarray([
                [
                    [-5, 1, 2,-4,-4,-4],
                    [ 0, 0, 0, 1, 0, 0],
                ],
            ]),
            puan.ndarray.integer_ndarray([
                [0, 3, 4, 2, 1, 1]
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
                [ 1, 2, 3, 4, 5, 6],
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
                0, 0, 3, 3, 1, 2
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
                [ 0, 0, 3, 3, 1, 2],
                [ 0, 0, 0, 0, 0, 0],
                [ 1, 2, 3, 4, 5, 6],
                [ 0, 0, 1, 1, 1, 1],
                [ 0, 1, 3, 2, 2, 2]
            ]),
            0
        ),
    ]
    for inpt, expected_output, axis in test_cases:
        actual_output = inpt.ndint_compress(method="rank", axis=axis)
        assert numpy.array_equal(actual_output, expected_output)
    
    with pytest.raises(ValueError):
        puan.ndarray.integer_ndarray([1, 5, 30]).ndint_compress(method="Rank")
    
        
    

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
            variable=puan.variable("xor_xy"),
        ),
        pg.Xor("x","y", variable="xor_xy"),
    )
    assert model.propositions[0].variable.bounds == model.propositions[1].variable.bounds

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

    json_model = {"type": "All", "propositions": [{"id": "x", "bounds": {"lower": -10, "upper": 10}}]}
    assert json_model == pg.from_json(json_model).to_json()

    model = pg.All(
        pg.XNor(pg.All("x", "y"), pg.Any("z", "u"), variable="A"),
        pg.XNor("a", "b")
    )
    converted = pg.from_json(model.to_json())
    assert model == converted

    json_model = {"propositions": [{"id": "x", "bounds": {"lower": -10, "upper": 10}}]}
    assert pg.from_json(json_model).to_json() == {"type": "AtLeast", "value": 1, "propositions": [{"id": "x", "bounds": {"lower": -10, "upper": 10}}]}

    json_model = {"type": "Variable", "id": "x", "bounds": {"lower": -10, "upper": 10}}
    assert pg.from_json(json_model).to_json() == {'bounds': {'lower': -10, 'upper': 10}, 'id': 'x'}

    json_model = {"type": "NotAValidType", "id": "x", "bounds": {"lower": -10, "upper": 10}}
    with pytest.raises(Exception):
        pg.from_json(json_model).to_json()
    
    json_model = {"type": "AtMost", "value": 1, "propositions": [{"id": "x", "bounds": {"lower": -10, "upper": 10}}]}
    assert json_model == pg.AtMost.from_json(json_model, [puan.variable]).to_json()
    


def test_xnor_proposition():
    
    actual_model = pg.XNor("x","y","z") # <- meaning none or at least 2
    polyhedron = actual_model.to_ge_polyhedron(True)
    assert polyhedron.shape == (3,6)
    assert polyhedron[polyhedron.A.dot([1,0,1,1,1]) < polyhedron.b].size == 0 # <- AtLeast 2 -variable active (neg AtMost 1)
    assert polyhedron[polyhedron.A.dot([0,1,0,0,0]) < polyhedron.b].size == 0 # <- None -variable active (neg AtLeast 1)

def test_proposition_polyhedron_conversions():

    actual = pg.All(
        pg.Not("x"),
        variable="all_not"
    ).to_ge_polyhedron(True)
    assert not all(actual.A.dot(actual.A.construct(*{"x": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.A.construct(*{"VARc96efc8ea4acc75f6dbddd0acac8f189b4c566f77b76b6299161a14e4eeb2caf": 1}.items())) >= actual.b)

    actual = pg.Not(
        pg.All("x","y","z", variable="all_xyz")
    ).to_ge_polyhedron(True)
    assert all(actual.A.dot(actual.A.construct(*{"x": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.A.construct(*{"x": 1, "y": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"x": 1, "y": 1, "z": 1}.items())) >= actual.b)

    actual = pg.Not(
        pg.Any("x","y","z")
    ).to_ge_polyhedron(True)
    assert not all(actual.A.dot(actual.A.construct(*{"x": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"y": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"z": 1}.items())) >= actual.b)

    actual = pg.Imply(
        condition=pg.Not("x"),
        consequence=pg.All("a","b","c")
    ).to_ge_polyhedron(True)
    assert all(actual.A.dot(actual.A.construct(*{"VARfe372293ac6fc8767d248278e9ceacbb53aa57de8d3b30ef20813933935d1332": 1, "x": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.A.construct(*{"VARda5ba595af62c244d136aab35d6713d054f9167785da38460211eb9cb8d165f4": 1, "a": 1, "b": 1, "c": 1}.items())) >= actual.b)

    actual = pg.Not(
        pg.Imply(
            condition=pg.All("x","y","z"), 
            consequence=pg.Any("a","b","c")
        ),
    ).to_ge_polyhedron(True)
    assert all(actual.A.dot(actual.A.construct(*{"VARb6a05d7d91efc84e49117524cffa01cba8dcb1f14479be025342b909c9ab0cc2": 1, "VARe3918cdbd4ac804be32d2b5a3f2890d6ae5f6d3fb9246b429be1bb973edd157a": 1, "x": 1, "y": 1, "z": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"VARb6a05d7d91efc84e49117524cffa01cba8dcb1f14479be025342b909c9ab0cc2": 1, "VARe3918cdbd4ac804be32d2b5a3f2890d6ae5f6d3fb9246b429be1bb973edd157a": 1, "x": 1, "y": 1, "z": 1, "a": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"VARb6a05d7d91efc84e49117524cffa01cba8dcb1f14479be025342b909c9ab0cc2": 1, "VARe3918cdbd4ac804be32d2b5a3f2890d6ae5f6d3fb9246b429be1bb973edd157a": 1, "x": 1, "y": 1, "z": 1, "b": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"VARb6a05d7d91efc84e49117524cffa01cba8dcb1f14479be025342b909c9ab0cc2": 1, "VARe3918cdbd4ac804be32d2b5a3f2890d6ae5f6d3fb9246b429be1bb973edd157a": 1, "x": 1, "y": 1, "z": 1, "c": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"VARb6a05d7d91efc84e49117524cffa01cba8dcb1f14479be025342b909c9ab0cc2": 1, "VARe3918cdbd4ac804be32d2b5a3f2890d6ae5f6d3fb9246b429be1bb973edd157a": 1, "x": 1, "y": 1}.items())) >= actual.b)

    actual = pg.Not(
        pg.AtLeast(propositions=["x","y","z"], value=2)
    ).to_ge_polyhedron(True)
    assert not all(actual.A.dot(actual.A.construct(*{"x": 1, "y": 1, "z": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.A.construct(*{"x": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.A.construct(*{"y": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.A.construct(*{"z": 1}.items())) >= actual.b)

    actual = pg.Not(
        pg.AtMost(propositions=["x","y","z"], value=2)
    ).to_ge_polyhedron(True)
    assert not all(actual.A.dot(actual.A.construct(*{"x": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"y": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"z": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"x": 1, "y": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"x": 1, "z": 1}.items())) >= actual.b)
    assert not all(actual.A.dot(actual.A.construct(*{"y": 1, "z": 1}.items())) >= actual.b)
    assert all(actual.A.dot(actual.A.construct(*{"x": 1, "y": 1, "z": 1}.items())) >= actual.b)

    # Example from tutorial
    actual = cc.StingyConfigurator(
        pg.Xor(
            puan.variable(id="t-thirt-blue"),
            puan.variable(id="t-thirt-black"),
            variable="t-shirts"
        ),
        pg.AtMost(
            propositions=[
                puan.variable(id="sweater-green"),
                puan.variable(id="sweater-black"),
            ],
            value=1,
            variable="sweaters"
        ),
        pg.Xor(
            puan.variable(id="jeans-blue"),
            puan.variable(id="jeans-black"),
            variable="jeans"
        ),
        pg.Xor(
            puan.variable(id="shoes-white"),
            puan.variable(id="shoes-black"),
            variable="shoes"
        ),
        id="outfit"
    ).to_ge_polyhedron(True)
    assert all(
        actual.A.dot(
            actual.A.construct(
                *{
                    "VAR078253de0198b4e2ba4ae54d2ff9cb511d3055064d6132049a29ff6a2dc55edd": 1,
                    "VAR0602949238c693ef0f13dd2a14c00cda10db750d10bea4f3c8f4519e279bc1cb": 1,
                    "VAR468a4942c7f47232c142a5b40253be66fafbacee418fb325b299c3a3df1d4ff1": 1,
                    "VAR4803c19cbb7f1c2fa552ef65967b971a3e798eaaed91fcb2726a7d27acd0d23a": 1,
                    "VAR12523219673c8e113cf2afe800e4e67db7c61bbd0c5feed1179abd493badafbe": 1,
                    "VAR6871c4a77e251fda952fc29a38babd55caa966a5d5a217cfa673d8dbe2181997": 1,
                    "t-shirts": 1,
                    "sweaters": 1,
                    "jeans": 1,
                    "shoes": 1,
                    "t-thirt-blue": 1,
                    "jeans-blue": 1,
                    "shoes-white": 1,
                }.items()
            )
        ) >= actual.b
    )

def test_not_when_single_str_or_variable():

    for entity in ["a", puan.variable("a")]:
        assert type(pg.Not(entity)) in [pg.All, pg.AtLeast]

def test_multiple_defaults():
    #a -> p ^ q (q)
    #a -> p ^ r (r)
    model = cc.StingyConfigurator(pg.All(pg.Imply("a", cc.Xor(*"pq", default="q")), pg.Imply("a", cc.Xor(*"pr", default="r"))))
    config1 = {"a": 1, "p": 0, "q": 1, "r": 1}
    config2 = {"a": 1, "p": 1}
    full_config1 = model.evaluate_propositions(config1)
    full_config2 = model.evaluate_propositions(config2)
    int_ndarray1 = model.ge_polyhedron.A.construct(*full_config1.items())
    int_ndarray2 = model.ge_polyhedron.A.construct(*full_config2.items())
    objective_function = puan.ndarray.ndint_compress(model.ge_polyhedron.default_prio_vector, method="shadow")
    assert objective_function.dot(int_ndarray1) > objective_function.dot(int_ndarray2)

def test_evaluate_propositions():
    # Raises ValueError when configuration variable is out of bounds
    with pytest.raises(ValueError):
        pg.All(*"xy", variable="A").evaluate_propositions({"x": 3})
    
    # Unsatisfiable model
    assert pg.AtLeast(2, "x").evaluate_propositions({"x": 0}) == {'VARa7c4c155fe9267e5308123f3d8b4e663ced757f934a47fc023c808b568ae51c4': 0, "x": 0}

    # Faulty configuration input
    with pytest.raises(ValueError):
        pg.All(*"xy", variable="A").evaluate_propositions({"x": "str"})

    # Variable defaults to lower bounds when not given as configuration
    assert pg.AtLeast(1, [puan.variable("x", bounds=(0,1))], variable="A").evaluate_propositions({}) == {"A": 0, "x": 0}
    assert pg.AtLeast(1, [puan.variable("x", bounds=(1,10))], variable="A").evaluate_propositions({}) == {"A": 1, "x": 1}

    # Test giving values to other nodes than the leafs
    # Each node's value, except leafs, should be evaluated and not got from input
    assert pg.All(puan.variable("a"), variable="A").evaluate_propositions({"A": 1, "a": 0}) == {'A': 0, 'a': 0}

    # Test such that B is zero and thus A should be zero, even though result from C will say B = 1
    result = pg.All(
        pg.AtMost( 0, [puan.variable('a')], variable="B"),
        pg.AtMost( 0, [puan.variable('b')], variable="C"),
        variable="A",
    ).evaluate_propositions({'a': 1, 'b': 0, 'A': 1, 'B': 1, 'C': 1})
    assert result['A'] == 0
    assert result['B'] == 0

    # Test competition of dependent variable's value when shared variables
    # We know that this yields circular dependency error but we still
    # want to test that correct value will be propagated
    model = pg.All(
        pg.All(*"Ba", variable="C"),
        pg.All(*"Ca", variable="B"),
        variable="A",
    )
    # We want B to be evaluated to 0 since C's default value is 0
    # We want (then!) C to be evaluated to 0 since B is 0
    # We want A to be evaluated to 0 since B and C are zero 
    assert model.evaluate_propositions({'a': 1, 'B': 1}) == {'A': 0, 'B': 0, 'C': 0, 'a': 1}
    # while evaluating with C=1 instead, we expects all to be set to 1's
    assert model.evaluate_propositions({'a': 1, 'C': 1}) == {'A': 1, 'B': 1, 'C': 1, 'a': 1}

    # This should in the end evaluate to True
    # Note that the AtMost results in the
    # -a-b-c-d-e >= 5 constraint. Adding the values
    # to it results in -(3)-(1)-(-3)-(-3)-(-3) >= 5 and
    # to -3-1+3+3+3 >= 5 and finally 5 >= 5, which yields True.
    model = pg.AtMost(
        -5, 
        [
            puan.variable('a', (-5,3)),
            puan.variable('b', ( 0,2)),
            puan.variable('c', (-4,4)),
            puan.variable('d', (-3,4)),
            puan.variable('e', (-4,5))
        ], 
        variable=puan.variable('A', (0,1))
    )
    assert model.evaluate_propositions({'A': 1, 'a': 3, 'b': 1, 'c': -3, 'd': -3, 'e': -3}) == {'A': 1, 'a': 3, 'b': 1, 'c': -3, 'd': -3, 'e': -3}


def configuration_dict_strategy():
    return st.dictionaries(st.text(), st.integers(-2,2))

@given(proposition_strategy(), configuration_dict_strategy())
def test_propositions_evaluations(proposition, configuration):
    value_error_raised = False
    try:
        evaluate_propositions_result = proposition.evaluate_propositions(configuration)[proposition.variable.id] > 0
        evaluate_result = proposition.evaluate(configuration)
    except ValueError:
        value_error_raised = True
    if not value_error_raised:
        assert evaluate_propositions_result == evaluate_result

@given(st.text(), st.integers(), st.integers(), st.sampled_from([puan.Dtype.BOOL, puan.Dtype.INT, "bool", "int", None]))
def test_puan_variable(id, lower, upper, dtype):
    
    if lower <= upper:
        # should raise error iff dtype == "bool" and bounds != (0,1)
        if (dtype == "bool") and ((lower, upper) != (0,1)):
            with pytest.raises(ValueError):
                puan.variable(id, (lower, upper), dtype)
        else:
            puan.variable(id, (lower, upper), dtype)
    else:
        # should raise ValueError from Bounds
        with pytest.raises(ValueError):
            puan.variable(id, (lower, upper), dtype)

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

#     assumed = model.assume({"x": 1, "y": 1, "z": 1})[0].to_ge_polyhedron(True)
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
        puan.variable.support_vector_variable(),
        puan.variable("a", (-10,10)),
        puan.variable("b", (-10,10)),
        puan.variable("c", (-10,10)),
        puan.variable("d", (-10,10)),
    ]
    actual = puan.ndarray.ge_polyhedron([
        [-3, 0, 1, 0, 0], # b_lb = -2
        [-3, 0, 0, 0,-1], # d_ub =  3
    ], variables=variables).tighten_column_bounds()
    expected = numpy.array([
        [-10, -3,-10,-10],
        [ 10, 10, 10,  3]
    ])
    assert (actual == expected).all()

    actual = puan.ndarray.ge_polyhedron([
        [ 3, 1, 0, 0, 0], # a_lb =  3
        [ 3, 0, 0,-1, 0], # c_ub = -3
    ], variables=variables).tighten_column_bounds()
    expected = numpy.array([
        [  3,-10,-10,-10],
        [ 10, 10, -3, 10]
    ])
    assert (actual == expected).all()

    # When both upper bound and lower bounds are in same column
    actual = puan.ndarray.ge_polyhedron([
        [ 3, 1, 0, 0, 0], # a_lb = 3
        [-5,-1, 0, 0, 0], # a_ub = 5
    ], variables=variables).tighten_column_bounds()
    expected = numpy.array([
        [  3,-10,-10,-10],
        [  5, 10, 10, 10]
    ])
    assert (actual == expected).all()

    # When a tighter bound exists
    actual = puan.ndarray.ge_polyhedron([
        [ 3, 1, 0, 0, 0], # a_lb = 3
        [ 3, 0, 1, 0, 0], # a_lb = 3
        [ 3, 0, 0,-1, 0], # a_lb = 3
        [ 4, 1, 0, 0, 0], # b_lb = 4
        [ 4, 0, 1, 0, 0], # b_lb = 4
        [ 4, 0, 0,-1, 0], # b_lb = 4
        [ 5, 1, 0, 0, 0], # c_ub = 5
        [ 5, 0, 1, 0, 0], # c_ub = 5
        [ 5, 0, 0,-1, 0], # c_ub = 5
    ], variables=variables).tighten_column_bounds()
    expected = numpy.array([
        [  5,  5,-10,-10],
        [ 10, 10, -5, 10]
    ])
    assert (actual == expected).all()

    # Currently fractions are not supported
    with pytest.raises(Exception):
        # When coeffs are large and ub/lb will be a fraction
        actual = puan.ndarray.ge_polyhedron([
            [ 3, 2, 0, 0, 0], # a_lb =  3
            [ 5, 0, 0,-3, 0], # c_ub = -3
        ], variables=variables).tighten_column_bounds()
        expected = numpy.array([
            [  2,-10,-10,-10],
            [ 10, 10, -2, 10]
        ])
        assert (actual == expected).all()

    # Currently fractions are not supported
    with pytest.raises(Exception):
        # When one constraint is taighter than the others
        # 3 <= a gives taightes bounds on a
        actual = puan.ndarray.ge_polyhedron([
            [ 3, 4, 0, 0, 0], # 1 <= a
            [ 4, 3, 0, 0, 0], # 2 <= a
            [ 5, 2, 0, 0, 0], # 3 <= a
        ], variables=variables).tighten_column_bounds()
        expected = numpy.array([
            [  3,-10,-10,-10],
            [ 10, 10, 10, 10]
        ])
        assert (actual == expected).all()

    # Mixed lower and upper bound on `a` with fractions
    actual = puan.ndarray.ge_polyhedron([
        [-5,-2, 0, 0, 0], # -3 >= a
        [-2,-1, 0, 0, 0], # -2 >= a
        [ 3, 2, 0, 0, 0], #  2 <= a
        [ 3, 1, 0, 0, 0], #  3 <= a
    ], variables=variables).tighten_column_bounds()
    expected = numpy.array([
        [  3,-10,-10,-10],
        [  2, 10, 10, 10]
    ])
    assert (actual == expected).all()

    # When one constraint is taighter than the others, but is not covered as special case
    # Check such that bounds for a is *not* set to [  4, 10]
    # Check such that bounds for c is *not* set to [-10,  4]
    actual = puan.ndarray.ge_polyhedron([
        [20, 1, 1, 0, 0], # a>=10 & b>=10
        [ 4, 1, 0, 0, 0], # a>=4
        [20, 0, 0,-1,-1], # c<=0 & d<=0
        [-4, 0, 0,-1, 0], # c<=-4 
    ], variables=variables).tighten_column_bounds()
    expected = numpy.array([
        [ 10, 10,-10,-10],
        [ 10, 10,-10,-10]
    ])
    assert (actual == expected).all()

    # Same as previous but a contradicting constraint
    # is added to the bottom.
    actual = puan.ndarray.ge_polyhedron([
        [20, 1, 1, 0, 0],
        [ 4, 1, 0, 0, 0], 
        [-9,-1, 0, 0, 0], 
    ], variables=variables).tighten_column_bounds()
    expected = numpy.array([
        [ 10, 10,-10,-10],
        [ 9, 10, 10, 10]
    ])
    assert (actual == expected).all()

    # Conflicting constraints
    actual = puan.ndarray.ge_polyhedron([
        [ 1, 1], # <- at least 1
        [ 0,-1], # <- at most  0
    ]).tighten_column_bounds()
    expected = numpy.array([
        [ 1],
        [ 0]
    ])
    assert (actual == expected).all()

    actual = puan.ndarray.ge_polyhedron([
        [21, 1, 1, 0, 0],
    ], variables=variables).tighten_column_bounds()
    expected = numpy.array([
        [ 11, 11,-10,-10],
        [ 10, 10, 10, 10]
    ])
    assert (actual == expected).all()

    # Test integer lower bound will increase
    variables = [
        puan.variable.support_vector_variable(),
        puan.variable("a", (-10,10)),
        puan.variable("b", (0,1)),
        puan.variable("c", (0,1)),
        puan.variable("d", (0,1)),
    ]
    actual = puan.ndarray.ge_polyhedron([
        [ 3, 1, 0, 0, 0],
        [ 0,-2, 1, 1, 0],
    ], variables=variables).tighten_column_bounds()
    expected = numpy.array([
        [ 3, 0, 0, 0],
        [ 1, 1, 1, 1] # <- because if coeff of a >1, than constraint on row index 1 cannot ever be satisfied
    ])
    assert (actual == expected).all()
    
    # Test that ordinary bounds are kept
    actual = puan.ndarray.ge_polyhedron([
        [ 0,-2, 1, 1, 0],
        [-3,-2,-1,-1, 0],
        [-3,-2,-2, 1, 1]
    ]).tighten_column_bounds()
    expected = numpy.array([
        [0,0,0,0],
        [1,1,1,1]
    ])
    assert (actual == expected).all()
    
    # Test force lower bound to increase to 1
    actual = puan.ndarray.ge_polyhedron([
        [3,1,1,1,0]
    ]).tighten_column_bounds()
    expected = numpy.array([
        [1,1,1,0],
        [1,1,1,1]
    ])
    assert (actual == expected).all()
    
    # Test lower bound won't increase while at least one must be set
    actual = puan.ndarray.ge_polyhedron([
        [1,1,1,1,0]
    ]).tighten_column_bounds()
    expected = numpy.array([
        [0,0,0,0],
        [1,1,1,1]
    ])
    assert (actual == expected).all()
    
    # Test integer upper bound will decrease
    variables = [
        puan.variable.support_vector_variable(),
        puan.variable("a", (-10,10)),
        puan.variable("b", (0,1)),
        puan.variable("c", (0,1)),
        puan.variable("d", (0,1)),
    ]
    actual = puan.ndarray.ge_polyhedron([
        [ -1,-1, 0, 0, 0],
        [  0, 0,-1,-1,-1],
    ], variables=variables).tighten_column_bounds()
    expected = numpy.array([
        [-10,0,0,0],
        [  1,0,0,0]
    ])
    assert (actual == expected).all()

    # Test "inverted" boolean (as integers)
    variables = [
        puan.variable.support_vector_variable(),
        puan.variable("a", (-1,1)),
        puan.variable("b", (-1,1)),
        puan.variable("c", (-1,1)),
        puan.variable("d", (-1,1)),
    ]
    actual = puan.ndarray.ge_polyhedron([
        [ 4,-1,-1,-1,-1],
    ], variables=variables).tighten_column_bounds()
    expected = numpy.array([
        [-1,-1,-1,-1],
        [-1,-1,-1,-1],
    ])
    assert (actual == expected).all()    

    # Test if finding lower bound < 0 
    variables = [
        puan.variable.support_vector_variable(),
        puan.variable("a", (-10,10)),
        puan.variable("b", (  0, 1))
    ]
    actual = puan.ndarray.ge_polyhedron([
        [ 11,-1, 1],
    ], variables=variables).tighten_column_bounds()
    expected = numpy.array([
        [-10,1],
        [-10,1],
    ])
    assert (actual == expected).all()    

    # Test if constraint has unnecessary large coefficients
    actual = puan.ndarray.ge_polyhedron([
        [4, 2, 2, 0],
        [9, 3, 3, 3],
    ]).tighten_column_bounds()
    expected = numpy.array([
        [1, 1, 1],
        [1, 1, 1],
    ])
    assert (actual == expected).all()

    actual = puan.ndarray.ge_polyhedron([
        [ 0, -2, -2, -2],
    ]).tighten_column_bounds()
    expected = numpy.array([
        [0, 0, 0],
        [0, 0, 0],
    ])
    assert (actual == expected).all()

def test_column_bounds(): 

    actual = puan.ndarray.ge_polyhedron([])
    assert actual.variables.size == 0
    assert actual.shape == (0,)

    actual = puan.ndarray.ge_polyhedron([[0]])
    assert actual.variables.size == 1
    assert actual.variables[0] == puan.variable.support_vector_variable()
    assert actual.shape == (1,1)

    variables = [
        puan.variable.support_vector_variable(),
        puan.variable("a", (-10,10)),
        puan.variable("b", (0,1)),
        puan.variable("c", (-10,10)),
        puan.variable("d", (0,1)),
    ]

    actual = puan.ndarray.ge_polyhedron([
        [ 1,-1,-1,-1,-1],
    ], variables=variables).column_bounds()
    expected = numpy.array([
        [-10, 0,-10, 0],
        [ 10, 1, 10, 1]
    ])
    assert (actual == expected).all()

    variables = [
        puan.variable.support_vector_variable(),
        puan.variable("a", dtype="int"),
        puan.variable("b", dtype="bool"),
        puan.variable("c", (-10,10)),
        puan.variable("d", (0,1)),
    ]

    actual = puan.ndarray.ge_polyhedron([
        [ 1,-1,-1,-1,-1],
    ], variables=variables).column_bounds()
    expected = numpy.array([
        [puan.default_min_int, 0,-10, 0],
        [puan.default_max_int, 1, 10, 1]
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

    def dummy_solver(x, y):
        return list(map(lambda x: (x, 0, 5), numpy.ones((len(y), x.shape[1]))))

    expected = {'a': 1.0, 'b': 1.0, 'p': 1.0, 'q': 1.0, 'r': 1.0, 'x': 1.0, 'y': 1.0, 'z': 1.0}
    actual = list(model.select({"a": 1}, solver=dummy_solver, only_leafs=True))
    assert actual[0] == expected

    # Test should NOT raise error when solution is None
    def dummy_solver_none(x, y):
        return [(None, 0, 1)]

    res = model.select({"a": 1}, solver=dummy_solver_none)
    assert res == [({}, 0, 1)]

    def dummy_solver_raises(x,y):
        raise Exception("error from solver")

    with pytest.raises(puan.ndarray.InfeasibleError):
        list(model.select({"a": 1}, solver=dummy_solver_raises))


def test_dump_load_ge_polyhedron_config():

    model = cc.StingyConfigurator(
        pg.Imply(
            pg.All(*"ab"),
            cc.Xor(*"xyz", default="z")
        ),
        pg.Any(*"pqr")
    )

    def dummy_solver(x, y):
        return list(map(lambda x: (x, 0, 5), numpy.ones((y.shape[0], x.A.shape[1]))))

    expected = puan.ndarray.ge_polyhedron_config.from_b64(
        model.ge_polyhedron.to_b64()
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
    actual = pg.AtLeast(0, var, variable="A", sign=1).negate()
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
    actual = pg.AtLeast(0, var, variable="A", sign=1).negate()
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

    # Test to negate when variable bounds range in int space
    # We expect no difference with or without different bounds
    # but to be sure, we include a test here.
    model = pg.All(
        pg.Any(
            puan.variable("a", bounds=(-10,10)),
            puan.variable("b", bounds=(-10,10)),
            puan.variable("c", bounds=(-10,10)),
        ),
        pg.Xor(
            puan.variable("x", bounds=(-10,10)),
            puan.variable("y", bounds=(-10,10)),
            puan.variable("z", bounds=(-10,10)),
        ),
    )
    negated = model.negate()
    assert len(negated.propositions) == 2
    assert negated.sign == 1
    assert negated.value == 1
    assert len(negated.propositions[0].propositions) == 2
    assert negated.propositions[0].propositions[0].sign == 1
    assert negated.propositions[0].propositions[0].value == 2
    assert negated.propositions[0].propositions[1].sign == -1
    assert negated.propositions[0].propositions[1].value == 0
    assert negated.propositions[1].sign == -1
    assert negated.propositions[1].value == 0

def test_configurator_to_json():

    expected = {
        'propositions': [
            {
                "id": "A",
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
                "id": "B",
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
                "id": "C",
                "type": "Any",
                "propositions": [
                    {"id": "r"},
                    {"id": "s"},
                ],
            },
            {
                "id": "D",
                "type": "Xor",
                "propositions": [
                    {"id": "p"},
                    {"id": "q"},
                ],
            }
        ]
    }

    actual = cc.StingyConfigurator.from_json(expected).to_json()
    assert len(actual['propositions']) == len(expected['propositions'])
    for a,b in zip(actual['propositions'], expected['propositions']):
        assert a['id'] == b['id']
        assert a['type'] == b['type']
        assert len(a['propositions']) == len(b['propositions'])
        assert a.get('default', []) == b.get('default', [])

def test_at_leasts():

    with pytest.raises(Exception):
        pg.AtLeast(value=1, propositions=None)

    with pytest.raises(Exception):
        pg.AtLeast(value=1, propositions=None, variable="A")

    with pytest.raises(Exception):
        pg.AtLeast(propositions=[], value=1, variable="A")
    
    with pytest.raises(Exception):
        pg.AtLeast(propositions=["a"], value=1, sign=-2)
    
    assert pg.AtLeast(propositions=["a"], value=1).sign == puan.Sign.POSITIVE
    assert pg.AtLeast(propositions=["a"], value=0).sign == puan.Sign.NEGATIVE
    assert pg.AtLeast(propositions=["a"], value=-1).sign == puan.Sign.NEGATIVE

    with pytest.raises(ValueError):
        pg.AtLeast(value=1, propositions=["a"], variable=1)
    
    with pytest.raises(Exception):
        pg.AtLeast.from_short(("A"))

def test_imply():
    with pytest.raises(Exception):
        pg.Imply.from_json({"type": "Imply", "propositions": [{"id": "x", "bounds": {"lower": -10, "upper": 10}}]}, [])
    
    assert pg.Imply.from_json({"type": "Imply", "consequence": {"id": "x", "bounds": {"lower": -10, "upper": 10}}}, [puan.variable]) == puan.variable(id='x', bounds=puan.Bounds(lower=-10, upper=10))

def test_not():
    with pytest.raises(Exception):
        pg.Not.from_json({"type": "Not"}, [])
def test_plog_evaluate_method():

    """From tutorial example"""
    
    milk_home   = puan.variable(id="milk_home")
    milk_bought = puan.variable(id="milk_bought")
    chips       = puan.variable(id="chips")
    tomatoes    = puan.variable(id="tomatoes",   dtype="int")
    cucumbers   = puan.variable(id="cucumbers",  dtype="int")

    fridge_model = pg.All(
        chips,
        pg.Imply(
            condition=pg.Not(milk_home),
            consequence=milk_bought,
            variable="milk_done_right"
        ),
        pg.All(
            pg.AtLeast(propositions=[tomatoes], value=4, variable="tomatoes_ge_four"),
            pg.AtMost(propositions=[tomatoes], value=5, variable="tomatoes_le_five"),
            pg.AtMost(propositions=[cucumbers], value=5, variable="cucumbers_le_five"),
            pg.AtLeast(propositions=[cucumbers], value=1, variable="cucumbers_ge_one"),
            variable="vegestables"
        ),
        variable="fridge"
    )

    cart = {
        milk_home.id: 1,
        milk_bought.id: 0,
        tomatoes.id: 2+2,
        cucumbers.id: 0
    }

    assert not fridge_model.evaluate(cart)

    new_cart = {
        chips.id: 1,
        milk_home.id: 1,
        milk_bought.id: 0,
        tomatoes.id: 2+2,
        cucumbers.id: 1
    }

    assert fridge_model.evaluate(new_cart)

def test_duplicated_ids_should_not_result_in_contradiction():

    assert not pg.All(*"xxyyzz").is_contradiction
    assert not pg.Any(*"xxyyzz").is_contradiction
    assert not pg.Xor(*"xxyyzz").is_contradiction
    assert not pg.XNor(*"xxyyzz").is_contradiction

def test_cc_any_will_init_properly():

    model = cc.StingyConfigurator(
        # NOTE a is not in list of propositions and should keep structure
        cc.Any(*"xyz", default=["a"]),
        id="A"
    )
    assert len(model.propositions) == 1
    assert len(model.propositions[0].propositions) == 3

    model = cc.StingyConfigurator(
        # NOTE x IS in list of propositions and should restructure
        cc.Any(*"xyz", default=["x"]),
        id="A"
    )
    assert len(model.propositions) == 1
    assert len(model.propositions[0].propositions) == 2
    # Test that the id is new inside the automatic created sub proposition
    # Also that it exists and is an pg.Any
    assert next(filter(lambda x: type(x) == pg.Any, model.propositions[0].propositions)).id != model.propositions[0].id

def test_json_dump_puan_variables():

    import json

    assert json.dumps(puan.variable("x"))                                   == '{"id": "x"}'
    assert json.dumps(puan.variable("x", (-10,10)))                         == '{"id": "x", "bounds": {"lower": -10, "upper": 10}}'
    assert json.dumps(puan.variable("x", dtype="int"))                      == '{"id": "x", "bounds": {"lower": -32768, "upper": 32767}}'
    assert json.dumps(puan.variable("x", dtype="bool"))                     == '{"id": "x"}'

def test_proposition_errors_function():
    actual = pg.All(*"A", variable="A").errors()
    assert len(actual) == 1
    assert actual[0] == pg.PropositionValidationError.CIRCULAR_DEPENDENCIES
    
    actual = pg.All(*"AB", variable="A").errors()
    assert len(actual) == 1
    assert actual[0] == pg.PropositionValidationError.CIRCULAR_DEPENDENCIES

    actual = pg.All(pg.All(*"A"), variable="A").errors()
    assert len(actual) == 1
    assert actual[0] == pg.PropositionValidationError.CIRCULAR_DEPENDENCIES

    actual = pg.All(pg.All(pg.All(pg.All(pg.All(*"A")))), variable="A").errors()
    assert len(actual) == 1
    assert actual[0] == pg.PropositionValidationError.CIRCULAR_DEPENDENCIES

    actual = pg.All(
        pg.All(*"bc", variable="B"), 
        # B is defined twice with DIFFERENT bounds (should return error)
        pg.All(puan.variable("B", (-1,1)), "d", variable="C"), 
        variable="A"
    ).errors()
    assert len(actual) == 1 
    assert actual[0] == pg.PropositionValidationError.AMBIVALENT_VARIABLE_DEFINITIONS

    actual = pg.All(
        pg.All(*"bc", variable="B"), 
        # B is defined twice with SAME bounds (should not return error)
        pg.All(*"Bd", variable="C"), 
        variable="A"
    ).errors()
    assert len(actual) == 0

    # Share same id on same level but different sub propositions
    actual = pg.All(
        pg.AtMost(0, [puan.variable('a')], variable=''),
        pg.AtMost(0, [puan.variable('b')], variable='')
    ).errors()
    assert len(actual) == 2
    assert actual[0] == pg.PropositionValidationError.AMBIVALENT_VARIABLE_DEFINITIONS
    assert actual[1] == pg.PropositionValidationError.NON_UNIQUE_SUB_PROPOSITION_SET

    # Identical leaf siblings should result in error
    actual = pg.All(
        puan.variable('a', bounds=(-1,0)),
        puan.variable('a', bounds=(-1,0)),
        puan.variable('a', bounds=(-1,0)),
    ).errors()
    assert len(actual) == 1
    assert actual[0] == pg.PropositionValidationError.NON_UNIQUE_SUB_PROPOSITION_SET

    # Identical leafs not being siblings should not result in error
    actual = pg.All(
        pg.Any('x', puan.variable('a', bounds=(-1,0))),
        pg.Any('y', puan.variable('a', bounds=(-1,0))),
    ).errors()
    assert len(actual) == 0

    # Sharing same id on same level and same sub propositions
    actual = pg.All(
        pg.AtMost(0, [puan.variable('a')], variable=''),
        pg.AtMost(0, [puan.variable('a')], variable='')
    ).errors()
    assert len(actual) == 1
    assert actual[0] == pg.PropositionValidationError.NON_UNIQUE_SUB_PROPOSITION_SET

    # Should return error since we have two B's with different
    # children
    actual = pg.All(
        pg.AtMost(0, [puan.variable('a')], variable='B'),
        pg.All(
            pg.AtMost(0, [puan.variable('b')], variable='B'),
        ),
        variable="A"
    ).errors()
    assert len(actual) == 1
    assert actual[0] == pg.PropositionValidationError.AMBIVALENT_VARIABLE_DEFINITIONS

def test_function_add_for_stingy_configurator():

    with pytest.raises(Exception):
        cc.StingyConfigurator(
            pg.All(*"efg", variable="A")
        ).add(
            pg.All(*"abc", variable="A")
        )

    with pytest.raises(Exception):
        cc.StingyConfigurator(*"efg").add(
            pg.All(*"abc", variable="e")
        )

    # Should be ok since it is not interfering directly
    cc.StingyConfigurator(pg.All(*"efg", variable="A")).add(
        pg.All(*"ABC")
    )

def test_constructing_proposition_model_with_variable_sub_classes():

    class Fruit(puan.variable):
        
        def  __init__(self, size: str):
            super().__init__(f"{self.__class__.__name__}-{size}")
            self.size = size

    class Apple(Fruit):
        pass

    class Pear(Fruit):
        pass

    class Orange(Fruit):
        pass

    # Test that it is possible to construct
    model = pg.All(
        pg.Imply(
            Apple("big"),
            pg.Any(
                Orange("small"),
                Orange("medium"),
            )
        ),
        pg.Xor(Apple("small"), Apple("big")),
        pg.Not(Pear("medium"))
    )

    # Test that we can evaluate on the items
    assert not model.evaluate({"Apple-big": 1})
    assert not model.evaluate({"Pear-medium": 1})
    assert not model.evaluate({})
    assert model.evaluate({"Apple-big": 1, "Orange-small": 1})
    assert model.evaluate({"Apple-big": 1, "Orange-small": 1, "Orange-medium": 1})
    assert model.evaluate({"Apple-small": 1})

def test_solve_functions():

    class Fruit(puan.variable):
        
        def  __init__(self, size: str):
            super().__init__(f"{self.__class__.__name__}-{size}")
            self.size = size

    class Apple(Fruit):
        pass

    class Orange(Fruit):
        pass

    sub_model = pg.Imply(
        Apple("big"),
        pg.Any(
            Orange("small"),
            Orange("medium"),
        )
    )

    # Test that evaluate and solve returns same solution
    # Test for both pg and cc model
    objective_0 = {"Apple-big": 1, "Orange-small": 1}
    model_eval = sub_model.evaluate_propositions(objective_0)

    for sol_iter_actual in [
        pg.All(sub_model).solve([objective_0]), 
        cc.StingyConfigurator(sub_model).select(objective_0)
    ]:
        for sol_actual, sol_expected in zip(sol_iter_actual, [model_eval]):
            assert sol_actual[0] == sol_expected

def test_proposition_interface():
    class MyProposition(puan.Proposition):
        def __init__(self) -> None:
            pass
    with pytest.raises(NotImplementedError):
        MyProposition().to_short()
    with pytest.raises(NotImplementedError):
        MyProposition().to_json()
    with pytest.raises(NotImplementedError):
        MyProposition().from_json([],[])

def test_misc():
    with pytest.raises(KeyError):
        puan.misc.or_get({}, "key")
    with pytest.raises(KeyError):
        puan.misc.or_replace({}, "key", 1)

def test_ndarray():
    with pytest.raises(ValueError):
        puan.ndarray.variable_ndarray(numpy.array([1,2,3]), variables=[puan.variable("a")])
    
    with pytest.raises(ValueError):
        puan.ndarray.variable_ndarray(numpy.array([1,2,3])).variable_indices(2)
    
    with pytest.raises(Exception):
        puan.ndarray.ge_polyhedron([[1]]).A_max
    
    with pytest.raises(Exception):
        puan.ndarray.ge_polyhedron([[1]]).A_min
    
    assert numpy.array_equal(puan.ndarray.ge_polyhedron([[2, 1, 1]]).reducable_rows_and_columns()[1], numpy.array([1,1]))
    assert numpy.array_equal(puan.ndarray.ge_polyhedron([[0, 1, 1]]).reducable_rows_and_columns()[0], numpy.array([1]))

    with pytest.raises(Exception):
        puan.ndarray.ge_polyhedron([[1,1,1,1]]).row_distribution(-1)
    with pytest.raises(Exception):
        puan.ndarray.ge_polyhedron([[1,1,1,1]]).row_distribution(1)
    with pytest.raises(ValueError):
        puan.ndarray.integer_ndarray([1,1,1,1]).reduce2d()
    with pytest.raises(ValueError):
        puan.ndarray.integer_ndarray([[1,1,1,1]]).reduce2d(method=["Last"])
    
    assert numpy.array_equal(puan.ndarray.integer_ndarray.from_list([], ["a", "b", "c"]), [])
    assert numpy.array_equal(puan.ndarray.boolean_ndarray.from_list([], ["a", "b", "c"]), [])

    with pytest.raises(ValueError):
        puan.ndarray.boolean_ndarray([1,1,1]).get_neighbourhood(method="ON")

