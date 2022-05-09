import abc
import ast
import maz
import functools
import itertools
import operator
import puan.misc as msc
import puan.ndarray as pnd
import puan
import numpy
import enum
import typing

default_min_int: int = numpy.iinfo(numpy.int16).min
default_max_int: int = numpy.iinfo(numpy.int16).max

class ge_constraint(tuple):

    def __new__(cls, instance) -> dict:
        return tuple.__new__(cls, instance)

class proposition(object):
    
    """
        A proposition is an abstract class and a logical object that can be resolved into a true or false value.

        Methods
        -------
        variables
            the variables inside this proposition
        to_constraints
            converts this proposition into a list of (greater-or-equal) constraints

    """
    
    @abc.abstractclassmethod
    def variables(self) -> typing.List[puan.variable]:

        """
            Returns variables for this proposition.

            Returns
            -------
                out : list : puan.variable
        """
        raise NotImplementedError()

    @abc.abstractclassmethod
    def to_constraints(self) -> typing.List[ge_constraint]:

        """
            Proposition as a (dict) ge-constraint. 

            Returns
            -------
                out : list : ge_constraint
        """
        raise NotImplementedError()

class variable_proposition(proposition):

    """
        A variable_proposition is a logical object that can be resolved into a true or false value.

        Methods
        -------
        variables
            the variables inside this proposition
        to_constraints
            converts this proposition into a list of (greater-or-equal) constraints
        representation
            is the string id representation of this proposition

    """

    def __init__(self, var: typing.Union[puan.variable, str], dtype: typing.Union[bool, int] = bool):
        if type(var) == str:
            var = puan.variable(var, bool)
        self.var = var

    @property
    def variables(self) -> typing.List[puan.variable]:
        return [self.var]

    def to_constraints(self, variable_predicate, min_int: int = default_min_int, max_int: int = default_max_int) -> typing.List[ge_constraint]:
        return []

    def representation(self) -> str:

        """
            How this variable proposition is represented as a string, defaulted to variable.id. 

            Examples
            --------
                >>> v = variable_proposition("x")
                >>> v.representation()
                >>> "x"

            Returns
            -------
                out : str
        """

        return self.var.id

    def __hash__(self):
        return self.var.__hash__()

    def __eq__(self, o) -> bool:
        return self.var == o.var

class boolean_variable_proposition(variable_proposition):

    """
        A boolean_variable_proposition is a logical object that can be resolved into a true or false value.
        The boolean_variable_proposition has a variable and a value it is expected to have.

    """

    def __init__(self, var: typing.Union[puan.variable, str], value: bool = True):
        super().__init__(var, bool)
        self.value = value

    def __repr__(self):
        return f"({self.var.__repr__()} = {self.value})"

    def __eq__(self, o) -> bool:
        return self.var == o.var and self.value == o.value

class discrete_variable_proposition(variable_proposition):

    """
        A discrete_variable_proposition is a logical object that can be resolved into a true or false value.
        The discrete_variable_proposition has a variable, an operator and a value. The variable dtype will
        be forced into an int and the value must be an int. The expression (x >= 1) is considered a
        discrete variable proposition.

    """

    def __init__(self, var: typing.Union[puan.variable, str], operator: str, value: int):
        if type(var) == str:
            var = puan.variable(var, int)
        super().__init__(var)
        self.operator = operator
        self.value = value

    def __hash__(self):
        return hash(self.var.id + self.operator + str(self.value))

    def __eq__(self, o) -> bool:
        return self.var == o.var and self.value == o.value and self.operator == o.operator

    @property
    def variables(self) -> typing.List[puan.variable]:
        return [self.var, self.supporting_variable()]

    def supporting_variable(self) -> str:

        """
            A discrete variable is a combination of two variables: the one given and one
            supporting variable. These two variables has a certain relation to one another.

            Examples
            --------
                >>> dv = discrete_variable_proposition(variable("x"), ">=", 3)
                >>> dv.supporting_variable()
                >>> "x>=3"

            Returns
            -------
                out : str
        """
        
        return puan.variable(self.var.id + self.operator + str(self.value), bool, True)

    def to_constraints(self, variable_predicate = lambda x: x, min_int: int = default_min_int, max_int: int = default_max_int) -> typing.List[ge_constraint]:
        if self.operator == ">=":
            return [
                (
                    [
                        variable_predicate(self.supporting_variable().id),
                        variable_predicate(self.var.id),
                        variable_predicate(0),
                    ],
                    [(max_int-self.value), -1, -self.value]
                ),
                (
                    [
                        variable_predicate(self.supporting_variable().id),
                        variable_predicate(self.var.id),
                        variable_predicate(0),
                    ],
                    [-self.value, 1, 0]
                )
            ]
        elif self.operator == "<=":
            return [
                (
                    [
                        variable_predicate(self.supporting_variable().id),
                        variable_predicate(self.var.id),
                        variable_predicate(0),
                    ],
                    [max_int, -1, self.value]
                ),
                (
                    [
                        variable_predicate(self.supporting_variable().id),
                        variable_predicate(self.var.id),
                        variable_predicate(0),
                    ],
                    [min_int+self.value, -1, min_int]
                )
            ]
        else:
            raise Exception(f"continuous variable has invalid operator: '{self.operator}'")

    def __repr__(self):
        return f"('{self.var.id}': {self.var.dtype}  {self.operator} {self.value})"

    def representation(self) -> str:
        return self.supporting_variable().id

class conditional_proposition(proposition):

    """
        A conditional_proposition is a logical object that can be resolved into a true or false value.
        The conditional_proposition has a relation and a list of propositions. There are two relation
        types (ALL/ANY) and the proposition will be considered true if either ALL or ANY of its propositions
        are true (depending if relation is ALL or ANY).

    """

    def __init__(self, relation: str, propositions: typing.List[typing.Union["conditional_propositions", variable_proposition]]):
        self.relation = relation
        self.propositions = propositions

    @property
    def variables(self) -> typing.List[puan.variable]:
        return list(set(itertools.chain(*map(operator.attrgetter("variables"), self.propositions))))

    def to_constraints(self, variable_predicate = lambda x: x, min_int: int = default_min_int, max_int: int = default_max_int) -> itertools.chain:
        return itertools.chain(*map(operator.methodcaller("to_constraints", variable_predicate=variable_predicate, min_int=min_int, max_int=max_int), self.propositions))

    def to_dnf(self) -> map:
        resolved = itertools.chain(
            map(
                conditional_proposition.to_dnf, 
                filter(lambda x: isinstance(x, conditional_proposition), self.propositions)
            ),
            map(
                lambda x: [(x,)],
                filter(lambda x: not isinstance(x, conditional_proposition), self.propositions)
            )
        )
        
        return itertools.starmap(
            maz.compose(list, itertools.chain),
            itertools.product(*resolved) if self.relation == "ALL" else resolved
        )

    def __repr__(self):
        join_on = [' & ', ' | ']
        return f"({join_on[self.relation == 'ANY'].join(map(operator.methodcaller('__repr__'), self.propositions))})"

    def __eq__(self, o) -> bool:
        return self.relation == o.relation and all(itertools.starmap(operator.eq, zip(self.propositions, o.propositions)))

class consequence_proposition(conditional_proposition):

    """
        A consequence_proposition is a logical object that can be resolved into a true or false value.
        The consequence_proposition is a conditional_proposition with the exception of an extra field 
        "default". This is used to mark which underlying propositions is default if many are considered
        equally correct.

    """

    def __init__(self, relation: str, propositions: typing.List[typing.Union["conditional_propositions", variable_proposition]], default: typing.List[variable_proposition] = []):
        super().__init__(relation, propositions)
        self.default = default

    def __repr__(self):
        return f"({super().__repr__()[:-1]}, {self.default})"

    def __eq__(self, o) -> bool:
        return super().__eq__(o) and self.default == o.default

class Implication(enum.Enum):

    ALL         = 0
    ANY         = 1
    XOR         = 2
    NONE        = 3
    MOST_ONE    = 4

    constants_map = {
        0: lambda n_cond, n_cons: (-n_cons,  1, n_cons-n_cons*n_cond),
        1: lambda n_cond, n_cons: (-n_cons,  1, -n_cons*n_cond+1),
        3: lambda n_cond, n_cons: (-n_cons, -1, -n_cond*n_cons),
        4: lambda n_cond, n_cons: (-n_cons, -1, -n_cons*n_cond-1),
    }

    def constant_functions(self) -> typing.List["Implication"]:
        constants_map = {
            0: lambda n_cond, n_cons: (-n_cons,  1, n_cons-n_cons*n_cond),
            1: lambda n_cond, n_cons: (-n_cons,  1, -n_cons*n_cond+1),
            3: lambda n_cond, n_cons: (-n_cons, -1, -n_cond*n_cons),
            4: lambda n_cond, n_cons: (-n_cons, -1, -n_cons*n_cond-1),
        }

        if self == Implication.XOR:
            return [constants_map[Implication.ANY.value], constants_map[Implication.MOST_ONE.value]]
        return [constants_map[self.value]]

    def constraint_values(self: "Implication", condition_indices: typing.List[int], consequence_indices: typing.List[int], support_variable_index: int = 0) -> zip:
        constant_functions = self.constant_functions()
        return zip(
            itertools.repeat(condition_indices + consequence_indices + [support_variable_index], len(constant_functions)),
            itertools.starmap(
                lambda cond_val, cons_val, support_val: list(
                    itertools.chain(
                        itertools.repeat(cond_val, len(condition_indices)),
                        itertools.repeat(cons_val, len(consequence_indices)),
                        itertools.repeat(support_val, 1),
                    )
                ),
                map(
                    lambda x: x(len(condition_indices), len(consequence_indices)),
                    constant_functions
                )
            )
        )

class implication_proposition(proposition):

    """
        A implication_proposition is a logical object that can be resolved into a true or false value.
        The implication_proposition has a logical structure of condition - implies -> consequence, or
        the more common sentence "if this then that". In other words, the proposition is false only if
        the condition is considered true while the consequence is false.

    """

    def __init__(self, implies: Implication, consequence: consequence_proposition, condition: conditional_proposition = conditional_proposition("ALL", [])):
        self.condition = condition
        self.consequence = consequence
        self.implies = implies

    @property
    def variables(self) -> typing.List[puan.variable]:
        return list(set(self.condition.variables + self.consequence.variables))

    def to_constraints(self, variable_predicate = lambda x: x, min_int: int = default_min_int, max_int: int = default_max_int) -> itertools.chain:
        varialble_predicate_map = maz.compose(
            list, 
            functools.partial(
                map, 
                maz.compose(
                    variable_predicate, 
                    operator.methodcaller("representation")
                )
            )
        )
        return itertools.chain(
            (
                itertools.chain(
                    *itertools.starmap(
                        maz.compose(list, self.implies.constraint_values),
                        itertools.product(
                            map(varialble_predicate_map, self.condition.to_dnf()), 
                            map(varialble_predicate_map, self.consequence.to_dnf())
                        )
                    )
                )
            ),
            self.condition.to_constraints(variable_predicate, min_int=min_int, max_int=max_int),
            self.consequence.to_constraints(variable_predicate, min_int=min_int, max_int=max_int)
        )

    @staticmethod
    def from_cicR(cicR: str) -> "implication_proposition":
        return implication_proposition(condition, consequence)

    def __repr__(self) -> str:
        return f"{self.condition.__repr__()} -[{self.implies.name}]> {self.consequence.__repr__()}"

    def __eq__(self, o) -> bool:
        return self.condition == o.condition and self.consequence == o.consequence and self.implies == o.implies

class conjunctional_proposition(conditional_proposition):

    """
        A conjunctional_proposition is a logical object that can be resolved into a true or false value.
        The conjunctional_proposition is a conditional_proposition with the relation type set to ALL.

    """

    def __init__(self, propositions: typing.List[conditional_proposition]):
        super().__init__(relation="ALL", propositions=propositions)

    @property
    def variables(self) -> typing.Set[puan.variable]:
        return set(itertools.chain(*map(operator.attrgetter("variables"), self.propositions)))

    def to_constraints(self, variable_predicate: lambda x: x, min_int=default_min_int, max_int=default_max_int) -> itertools.chain:
        return itertools.chain(*map(operator.methodcaller("to_constraints", variable_predicate=variable_predicate, min_int=min_int, max_int=max_int), self.propositions))

    def to_polyhedron(self, variable_predicate = None, support_variable_index: int = 0, integer_bounds: tuple = (default_min_int, default_max_int)) -> pnd.ge_polyhedron:

        """
            Converts into a ge_polyhedron.

            Notes
            -----
            Currently, only implication_proposition's are supported in list of propositions when converting.

            Returns
            -------
                out : ge_polyhedron

            Examples
            --------
                >>> cc.conjunctional_proposition([
                ...     cc.implication_proposition(
                ...         cc.Implication.XOR,
                ...         cc.consequence_proposition("ALL", [
                ...             cc.boolean_variable_proposition("x"),
                ...             cc.boolean_variable_proposition("y"),
                ...             cc.boolean_variable_proposition("z")
                ...         ]),
                ...         cc.conditional_proposition("ALL", [
                ...             cc.discrete_variable_proposition("m", ">=", 3),
                ...         ])
                ...     ),
                ... ]).to_polyhedron(integer_bounds=(0, 52))
                ge_polyhedron([[   -4,     0,    -3,    -1,    -1,    -1],
                               [   -3,    -1, 32764,     0,     0,     0],
                               [   -2,     0,    -3,     1,     1,     1],
                               [    0,     1,    -3,     0,     0,     0]], dtype=int16)

        """

        variables = sorted(self.variables)
        if variable_predicate is None:
            variables_repr = [support_variable_index] + list(map(operator.attrgetter("id"), variables))
            variable_predicate = variables_repr.index

        constraints = list(self.to_constraints(variable_predicate, *integer_bounds))
        constraints_unique = list(dict(zip(map(str, constraints), constraints)).values())
        matrix = numpy.zeros((len(constraints_unique), len(variables)+1), dtype=numpy.int16)
        for i, (indices, values) in enumerate(constraints_unique):
            matrix[i, indices] = values

        return pnd.ge_polyhedron(numpy.unique(matrix, axis=0), variables)


class cicR(tuple):

    """
        cicR is a data type of the condition - implies - consequence format. It is a
        tuple and is written as (condition, implies, consequence). 

    """

    implication_mapping = {
        "REQUIRES_ALL": Implication.ALL, 
        "REQUIRES_ANY": Implication.ANY, 
        "REQUIRES_EXCLUSIVELY": Implication.XOR, 
        "FORBIDS_ALL": Implication.NONE, 
        "ONE_OR_NONE": Implication.MOST_ONE, 
    }

    def __new__(cls, instance):
        return tuple.__new__(cls, instance)

    def to_implication_proposition(self) -> implication_proposition:

        """
            Converts into an implication_proposition -class

            Returns
            -------
                out : implication_proposition

            Examples
            --------
                >>> cc = cicR.from_string("(('a','b')),'REQUIRES_ALL',('x','y','z')")
                >>> cc.to_implication_proposition()
                (('a': <class 'bool'>  = True) & ('b': <class 'bool'>  = True)) -[ALL]> ((('x': <class 'bool'>  = True) & ('y': <class 'bool'>  = True) & ('z': <class 'bool'>  = True), [])
        """
        condition_prop = cicR._condition_to_conditional_proposition(self[0])
        return implication_proposition(
            cicR.implication_mapping[self[1]], 
            consequence_proposition(
                "ALL", 
                list(
                    map(
                        lambda variable: discrete_variable_proposition(variable[0], variable[1], variable[2]) if isinstance(variable, tuple) else boolean_variable_proposition(variable),
                        self[2]  
                    )
                ),
                default=list(
                    map(
                        boolean_variable_proposition,
                        self[3] if len(self) == 4 else []   
                    )
                )
            ),
            condition_prop if isinstance(condition_prop, conditional_proposition) else conditional_proposition("ALL", [condition_prop])
        )

    @staticmethod
    def _condition_to_conditional_proposition(condition) -> conditional_proposition:
        if len(condition) == 3 and list(condition)[1] in [">=", "<="]:
            return discrete_variable_proposition(*condition)
        elif type(condition) == str:
            return boolean_variable_proposition(condition)
        else:
            return conditional_proposition(
                "ALL" if isinstance(condition, tuple) or isinstance(condition, set) else 'ANY', 
                list(
                    map(
                        cicR._condition_to_conditional_proposition,
                        condition
                    )
                )
            )

    @staticmethod
    def from_string(string: str) -> "cicR":
        return cicR(ast.literal_eval(string))


class cicJE(dict):

    """
        cicJE is a data type of the condition - implies - consequence format. It is a
        dict and must have the properties "condition" and "consquence". 

    """

    def __new__(cls, instance):
        return dict.__new__(cicJE, instance)

    def to_implication_proposition(self, id_ident: str = "id") -> implication_proposition:

        """
            Converts into an implication_proposition -class

            Parameters
            ----------
                id_ident : str = "id"
                    the id-property in component objects.

            Returns
            -------
                out : implication_proposition

            Examples
            --------
                >>> r = cc.cicJE({
                ...     "condition": {},
                ...     "consequence": {
                ...         "ruleType": "REQUIRES_ALL",
                ...         "components": [
                ...             {"id": "x"},
                ...             {"id": "y"}
                ...         ]
                ...     }
                ... })
                >>> r.to_implication_proposition()
                () -[ALL]> ((('x': <class 'bool'>  = True) & ('y': <class 'bool'>  = True), [])
        """

        map_component = lambda component: discrete_variable_proposition(component[id_ident], component['operator'], component['value']) if ('operator' in component and 'value' in component) else boolean_variable_proposition(component[id_ident])
        return implication_proposition(
            implies=cicR.implication_mapping.get(self['consequence']['ruleType']), 
            consequence=consequence_proposition(
                "ALL", 
                list(
                    map(
                        map_component,
                        self['consequence'].get('components', [])
                    )
                ), 
                list(
                    map(
                        map_component, 
                        self['consequence'].get('default', [])
                    )
                )
            ), 
            condition=conditional_proposition(
                self.get('condition', {}).get('relation', 'ALL'), 
                list(
                    map(
                        lambda sub_condition: conditional_proposition(
                            sub_condition.get('relation', 'ALL'), 
                            list(
                                map(
                                    map_component,
                                    sub_condition.get('components', [])
                                )
                            )
                        ),
                        self.get('condition', {}).get('subConditions', [])
                    )
                )
            )
        )

    def rule_type_get(self) -> str:
        return self['consequence'].get('ruleType')

    def variables(self, id_ident: str = "id") -> set:

        """
            Return variables as a set from this cicJE.

            Parameters
            ----------
                id_ident : str
                    the id-property in component objects.

            Returns
            -------
                out : Set[str]
        """
        return set().union(
            cicJE._condition2variables(self, id_ident),
            cicJE._obj_variables(self['consequence'], id_ident)
        )

    @staticmethod
    def _obj_variables(obj, id_ident: str = "id") -> typing.Set[str]:
        return set(map(lambda y: y[id_ident], obj.get('components', [])))

    def _condition2variables(self, id_ident: str = "id") -> typing.Set[str]:
        return set().union(
            *map(
                maz.pospartial(
                    cicJE._obj_variables, 
                    [(1, id_ident)]
                ),
                self.get('condition', {}).get('subConditions', [])
            )
        )


class cicJEs(list):

    """
        A conjunction of cicJE's.

        Methods
        -------
        compress
            TODO
        split
            Splits a ruleset into subsets of independet rules.
        variables
            Return variables as a set from this list of cicJE's.
        to_cicRs
            Converts directly to cicRs data type (cicE data types in between).
        to_cicEs
            Converts to cicEs data type.
    """

    def to_conjunctional_proposition(self, id_ident: str = "id") -> conjunctional_proposition:
        
        """
            Converts into an conjunctional_proposition -class

            Parameters
            ----------
                id_ident : str = "id"
                    the id-property in component objects.

            Returns
            -------
                out : conjunctional_proposition
        """
        return conjunctional_proposition(
            list(
                map(
                    cicJE.to_implication_proposition, 
                    self
                )
            )
        )

    @staticmethod
    def _merge_consequences(rule1: dict, rule2: dict, id_ident: str = "id") -> tuple:
        """
        Merges the consequences of rule2 with rule1 if the rule logic for both rules will be preserved,
        i.e. if the condition is the same and the rule type allows for maintenability of the rule logic.
        """
        rule_merged = False
        def sort_subcondition_components(subconditions: list):
            for subcondition in subconditions:
                subcondition['components'] = sorted(subcondition['components'], key=lambda x: x[id_ident])

        # Performing this operation here will mean that we will sort already sorted rules from time to time.
        # Better way to handle this to increase performance?
        sort_subcondition_components(msc.or_get(rule1['condition'], ['subConditions', 'sub_conditions'], []))
        sort_subcondition_components(msc.or_get(rule2['condition'], ['subConditions', 'sub_conditions'], []))

        if cicJE.rule_type_get(rule1) == cicJE.rule_type_get(rule2) and rule1['condition'] == rule2['condition']:
            for component in rule2['consequence']['components']:
                if not component in rule1['consequence']['components']:
                    rule1['consequence']['components'].append(component)
            rule_merged = True
        return (rule1, rule_merged)

    @staticmethod
    def _rule_can_be_merged(rule: dict) -> bool:
        could_be_merged = {'REQUIRES_ALL': True,
                            'FORBIDS_ALL': True,
                            'PREFERRED': True,
                            'ONE_OR_NONE': False,
                            'REQUIRES_EXCLUSIVELY': False,
                            'REQUIRES_ANY': False}
        rule_type = lambda x: "REQUIRES_ALL" if (cicJE.rule_type_get(x) in ['REQUIRES_EXCLUSIVELY', 'REQUIRES_ANY'] and len(x['consequence']['components']) <= 1) else cicJE.rule_type_get(x)
        rule_type_key = 'ruleType' if 'ruleType' in rule['consequence'].keys() else 'rule_type'
        rule['consequence'][rule_type_key] = rule_type(rule)
        return could_be_merged[rule_type(rule)]

    def compress(self: list, id_ident: str= "id") -> "cicJEs":
        compressed_ruleset = []
        while self:
            current_rule = self.pop()
            rules_to_remove = []
            if not cicJEs._rule_can_be_merged(current_rule):
                compressed_ruleset.append(current_rule)
                continue
            # Search remaining rules to merge with current rule
            for rule in self:
                if not cicJEs._rule_can_be_merged(rule):
                    rules_to_remove.append(rule)
                    compressed_ruleset.append(rule)
                    continue
                merged = False
                current_rule, merged = cicJEs._merge_consequences(rule1=current_rule, rule2=rule, id_ident=id_ident)
                if merged:
                    rules_to_remove.append(rule)
            compressed_ruleset.append(current_rule)
            self = [rule for rule in self if not rule in rules_to_remove]

        return cicJEs(compressed_ruleset)

    def split(self: list, id_ident: str="id") -> typing.List["cicJEs"]:
        """
            Splits a ruleset into subsets of independet rules, i.e. the configuration can be solved for each ruleset separately.

            Parameters
            ----------
            id_ident : str
                the id-property in component objects.

            Returns
            -------
                out : List[List(cucJEs)]
        """
        rule_indices = list(range(len(self)))
        unexamined_rules = [rule_indices.pop(0)]
        examined_rules = []
        rules_in_relation_list = []
        while unexamined_rules:
            currently_examined_rule_index = unexamined_rules.pop()
            rule_type = cicJE.rule_type_get(self[currently_examined_rule_index])
            for i in rule_indices:
                if rule_type in ["REQUIRES_ALL", "REQUIRES_ANY", "PREFERRED"] and cicJE.rule_type_get(self[i]) in ["REQUIRES_ALL", "REQUIRES_ANY", "PREFERRED"] or\
                    (rule_type == "FORBIDS_ALL" and cicJE.rule_type_get(self[i]) == "FORBIDS_ALL"):
                    if set(cicJE.variables(self[currently_examined_rule_index], id_ident)) & set(cicJE._condition2variables(self[i]['condition'], id_ident)) or\
                        set(cicJE._condition2variables(self[currently_examined_rule_index]['condition'], id_ident)) & set(cicJE.variables(self[i], id_ident)):
                        unexamined_rules.append(i)
                else:
                    if set(cicJE.variables(self[currently_examined_rule_index], id_ident)) & set(cicJE.variables(self[i], id_ident)):
                        unexamined_rules.append(i)
            rule_indices = [i for i in rule_indices if not i in unexamined_rules]
            if currently_examined_rule_index not in examined_rules:
                examined_rules.append(currently_examined_rule_index)
            if not unexamined_rules:
                rules_in_relation_list.append(examined_rules)
                examined_rules = []
                if rule_indices:
                    unexamined_rules = [rule_indices[0]]
                    rule_indices.pop(0)

        return [cicJEs([self[index] for index in indices]) for indices in rules_in_relation_list]

    def variables(self, id_ident: str = "id") -> set:

        """
            Return variables as a set from this list of cicJE's.

            Parameters
            ----------
                id_ident : str
                    the id-property in component objects.

            Returns
            -------
                out : Set[str]
        """

        return set(
            itertools.chain(
                *map(
                    functools.partial(
                        cicJE.variables,
                        id_ident=id_ident,
                    ),
                    self
                )
            )
        )
