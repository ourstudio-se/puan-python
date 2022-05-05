import abc
import ast
import maz
import functools
import itertools
import operator
import puan.misc as msc
import puan
import numpy
import enum
import typing

class ge_constraint(tuple):

    def __new__(cls, instance) -> dict:
        return tuple.__new__(cls, instance)

class proposition(object):
    
    @abc.abstractclassmethod
    def variables(self) -> typing.List[puan.variable]:

        """
            Returns variables for this proposition
        """
        raise NotImplementedError()

    @abc.abstractclassmethod
    def to_constraints(self) -> typing.List[ge_constraint]:

        """
            Proposition as a (dict) ge-constraint.

            Return:
                ge_constraint
        """
        raise NotImplementedError()

class variable_proposition(proposition):

    def __init__(self, var: typing.Union[puan.variable, str], dtype: typing.Union[bool, int] = bool):
        if type(var) == str:
            var = puan.variable(var, bool)
        self.var = var

    def variables(self) -> typing.List[puan.variable]:
        return [self.var]

    def to_constraints(self, variable_predicate) -> typing.List[ge_constraint]:
        return []

    def representation(self) -> str:
        return self.var.id

    def __hash__(self):
        return self.var.__hash__()

class boolean_variable_proposition(variable_proposition):

    def __init__(self, var: typing.Union[puan.variable, str], value: bool = True):
        super().__init__(var, bool)
        self.value = value

    def __repr__(self):
        return f"{self.var.__repr__()} = {self.value}"

class discrete_variable_proposition(variable_proposition):

    def __init__(self, var: typing.Union[puan.variable, str], operator: str, value: int):
        if type(var) == str:
            var = puan.variable(var, int)
        super().__init__(var)
        self.operator = operator
        self.value = value

    def __hash__(self):
        return hash(self.var.id + self.operator + str(self.value))

    def variables(self) -> typing.List[puan.variable]:
        return [self.var, self.supporting_variable()]

    def supporting_variable(self):
        return puan.variable(self.var.id + self.operator + str(self.value), bool, True)

    def to_constraints(self, variable_predicate = lambda x: x, min_int: int = numpy.iinfo(numpy.int16).min, max_int: int = numpy.iinfo(numpy.int16).max) -> typing.List[ge_constraint]:
        if self.operator == ">=":
            return [
                (
                    [
                        variable_predicate(self.supporting_variable().id),
                        variable_predicate(self.var.id),
                        variable_predicate(0),
                    ],
                    [max_int, -1, -self.value]
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
                    [-min_int, -1, -(min_int-self.value)]
                )
            ]
        else:
            raise Exception(f"continuous variable has invalid operator: '{self.operator}'")

    def __repr__(self):
        return f"('{self.var.id}': {self.var.dtype}  {self.operator} {self.value})"

    def representation(self) -> str:
        return self.supporting_variable().id

class conditional_proposition(proposition):

    def __init__(self, relation: str, propositions: typing.List[typing.Union["conditional_propositions", variable_proposition]]):
        self.relation = relation
        self.propositions = propositions

    def variables(self) -> typing.List[puan.variable]:
        return list(set(itertools.chain(*map(operator.methodcaller("variables"), self.propositions))))

    def to_constraints(self, variable_predicate = lambda x: x) -> itertools.chain:
        return itertools.chain(*map(operator.methodcaller("to_constraints", variable_predicate=variable_predicate), self.propositions))

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

class consequence_proposition(conditional_proposition):

    def __init__(self, relation: str, propositions: typing.List[typing.Union["conditional_propositions", variable_proposition]], default: typing.List[variable_proposition] = []):
        super().__init__(relation, propositions)
        self.default = default

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

    def __init__(self, implies: Implication, consequence: consequence_proposition, condition: conditional_proposition = conditional_proposition("ALL", [])):
        self.condition = condition
        self.consequence = consequence
        self.implies = implies

    def variables(self) -> typing.List[puan.variable]:
        return list(set(self.condition.variables() + self.consequence.variables()))

    def to_constraints(self, variable_predicate = lambda x: x) -> itertools.chain:
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
            self.condition.to_constraints(variable_predicate),
            self.consequence.to_constraints(variable_predicate)
        )

    @staticmethod
    def from_cicR(cicR: str) -> "implication_proposition":
        return implication_proposition(condition, consequence)

class conjunctional_proposition(conditional_proposition):

    def __init__(self, propositions: typing.List[conditional_proposition]):
        super().__init__(relation="ALL", propositions=propositions)

    def variables(self) -> typing.Set[puan.variable]:
        return set(itertools.chain(*map(operator.methodcaller("variables"), self.propositions)))

    def to_constraints(self, variable_predicate: lambda x: x) -> itertools.chain:
        return itertools.chain(*map(operator.methodcaller("to_constraints", variable_predicate=variable_predicate), self.propositions))

    def to_ge_polytope(self, variable_predicate = None, support_variable_index: int = 0) -> puan.ge_polyhedron:
        variables = sorted(self.variables())
        variables_repr = [support_variable_index] + list(map(operator.attrgetter("id"), variables))
        if variable_predicate is None:
            variable_predicate = variables_repr.index

        constraints = list(self.to_constraints(variable_predicate))
        constraints_unique = list(dict(zip(map(str, constraints), constraints)).values())
        matrix = numpy.zeros((len(constraints_unique), len(variables)+1), dtype=numpy.int16)
        for i, (indices, values) in enumerate(constraints_unique):
            matrix[i, indices] = values

        return puan.ge_polyhedron(numpy.unique(matrix, axis=0), variables)

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

    def variables(self, id_ident: str = "id") -> list:

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

        return list(
            set(
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
        )
