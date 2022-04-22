import ast
import itertools
import operator
import typing 

"""
    # Condition-Implies-Consequence (cic)

    Condition-Implies-Consequence (cic) is an abstract data type defining logical relationship
    between variables in a combinatorial optimization manner. A cic is created from "if this then that"
    sentence and is easy to understand and grasp. It is also a specific instance from a propositional
    logic expression with the implies-operator in between a "if" and "then". 
    For example, "if it is raining then I'll take the umbrella" could be written as "a -> b" where 
    if a = "it is raining" and b = "take the umbrella".

    Data types:
        - cicR:     The RAW format of a cic, meaning the condition and consequence are both conjunctions.
                    This format has a one-to-one mapping into a linear programming constraint.
        - cicE:     A more Expressive format where the condition can be written either as a DNF or a CNF.
                    also "REQUIRES_EXCLUSIVELY" rule type exist here.
        - cicEs:    String version of cicE

"""

class cicR(list):
    """
        Condition-Implies-Consequence RAW (cicR) is a variant of cic logical rule defining relationship
        between variables. It consist of four parameters: condition, rule type, consequence
        and preferred. The data type is a list requiring at least three of these four parameters. The RAW
        stands for the most atomic level of a cic, meaning "and" is implied in condition and consequence.

        Parameters:
            condition: a set of variables (strings) where "and" -relation is implied between the variables.
            rule_type: enum/str: REQUIRES_ALL, REQUIRES_ANY, ONE_OR_NONE, FORBIDS_ALL
            consequence: a set of variables (strings) where "and" -relation is implied between the variables.
            preferred: when ambivalence exists, this parameter tells which variable to prefer over the others
    """
    pass

class cicRs(list):

    """
        Is a conjunction-list of `cicR` -items. 
    """
    _rule_conjunction_constants_map = {
        "REQUIRES_ALL": lambda n_cond, n_cons: (-n_cons, 1, n_cons-n_cons*n_cond), # reqall
        "REQUIRES_ANY": lambda n_cond, n_cons: (-n_cons, 1, -n_cons*n_cond+1), # reqany
        "FORBIDS_ALL":  lambda n_cond, n_cons: (-n_cons, -1, -n_cond*n_cons), #forball
        "ONE_OR_NONE":  lambda n_cond, n_cons: (-n_cons, -1, -n_cons*n_cond-1), # one-or-none
    }

    def to_value_map(self: typing.List[cic], variable_map: callable, support_variable_index: int = 0) -> dict:

        """
            Converts a list of cicR's into a value map.

            Parameters:
                self:                   : list = a cicR-list
                variable_map            : dict = a dict mapping variables to integers
                support_variable_index  : int  = index representing the support variable in a ge_polytope
            
            Return:
                dict: value_map
        """

        value_map = {}
        for i, (condition_col_idxs, rule_type, consequence_col_idxs) in enumerate(
            zip(
                map(variable_map, filter(operator.itemgetter(0), self)),
                filter(operator.itemgetter(1), self),
                map(variable_map, filter(operator.itemgetter(2), self))
            )
        ):
            (condition_constant, consequence_constant, support_constant) = self._rule_conjunction_constants_map[rule_type](
                len(condition_col_idxs),
                len(consequence_col_idxs),
            )
            if condition_constant != 0 and condition_col_idxs:
                value_map.setdefault(condition_constant, [[], []])
                value_map[condition_constant][0] += list(itertools.repeat(i, len(condition_col_idxs)))
                value_map[condition_constant][1] += condition_col_idxs

            if consequence_constant != 0:
                value_map.setdefault(consequence_constant, [[], []])
                value_map[consequence_constant][0] += list(itertools.repeat(i, len(consequence_col_idxs)))
                value_map[consequence_constant][1] += consequence_col_idxs

            if support_constant != 0:
                value_map.setdefault(support_constant, [[],[]])
                value_map[support_constant][0] += [i]
                value_map[support_constant][1] += [support_variable_index]

        return value_map

class cicE(list):

    """
        Condition-Implies-Consequence EXPRESSIVE (cicE) is a variant of cic logical rule defining relationship
        between variables. It consist of four parameters: condition, rule type, consequence
        and preferred. The data type is a list requiring at least three of these four parameters. The E
        stands for a expressive variant of a cic, meaning 
            - the condition can be formed as combinations of "or" and "and" relations
            - the rule types are extended with "REQUIRES_EXCLUSIVELY"
    """

    _rule_map: dict = {
        "REQUIRES_EXCLUSIVELY": ["REQUIRES_ANY", "ONE_OR_NONE"]
    }

    def _expload_condition(self) -> typing.List[cicE]:

        """
            Converts data type condition into a DNF.
            Argument `condition` is either a tuple of tuples and/or lists
            or list of tuples and/or lists.

            Example: 
                input: ((a & b) | (c & d))
                output: [[a,b], [c,d]]

            Return:
                List[Set[str]]
        """
        def collect_allany_variables(sub_conditions, all_type = tuple, any_type = list):
            all_variables_set = set()
            any_variables_chunks = []
            for sub_condition in sub_conditions:
                sub_type = type(sub_condition)
                variables = list(sub_condition)
                if sub_type == all_type:
                    all_variables_set.update(variables)
                elif sub_type == any_type:
                    any_variables_chunks.append(variables)
                elif sub_type == str:
                    all_variables_set.update(["".join(variables)])
                else:
                    raise Exception(f"got invalid type of condition chunk: `{sub_type}`")

            return all_variables_set, any_variables_chunks

        if len(condition) == 0:
            result = [{}]
        else:
            condition_type = type(condition)
            if condition_type == tuple:
                all_variables_set, any_variables_chunks = collect_allany_variables(condition)
                result = list(
                    itertools.starmap(
                        set.union,
                        zip(
                            itertools.cycle([all_variables_set]),
                            map(
                                puancore.comp.compose(list, set),
                                itertools.product(*any_variables_chunks)
                            )
                        )
                    )
                )

            elif condition_type == list:
                all_variables_set, any_variables_chunks = collect_allany_variables(
                    condition,
                    all_type=list,
                    any_type=tuple
                )
                result = list(
                    map(
                        lambda x: set(x) if type(x) != str else set([x]),
                        operator.add(
                            list(all_variables_set),
                            any_variables_chunks
                        )
                    )
                )
            
            elif condition_type == str:
                result = [{condition}]
            else:
                raise Exception(f"condition must be wrapped with `( )` or `[ ]` but found: {condition}")

        return list(
            map(
                lambda cnd: [
                    cnd, self[1], self[2], self[3] if len(self) == 4 else []
                ],
                result
            )
        )

    def _expload_rule_type(self) -> typing.List[cicE]:
        """
            Exploads rule types into many rules satisfying
            cicR's rule types.

            Example:
                input: [..., "REQUIRES_EXCLUSIVELY", ...]
                output: [
                    [..., "REQUIRES_ANY", ...],
                    [..., "ONE_OR_NONE", ...],
                ] 

            Return:
                list[cicE]
        """
        if self[1] in cicRs._rule_conjunction_constants_map:
            return list(
                map(
                    lambda x: [
                        self[0], x, self[1], self[2] if len(self) == 4 else []
                    ],
                    cicE._rule_map.get(self[1])
                )
            )

        return [self]
    
    def to_cicRs(self) -> typing.List[cicR]:

        """
            Converts a cicE into a list of cicRs.

            Return:
                list[cicR]
        """
        return list(
            itertools.chain(
                *map(
                    cicE._expload_rule_type,
                    itertools.chain(*self._expload_condition())
                )
            )
        )
    
    @staticmethod
    def from_string(cicE_str: str) -> "cicE":

        """
            Convert from string into a cicE.

            Example:
                input: "(('a','b'),'REQUIRES_EXCLUSIVELY',('x','y','z'), ('y',))"
                output: [("a", "b"), "REQUIRES_EXCSLUIVELY", ("x", "y", "z"), ("y")]

            Return:
                cicE
        """
        return cicE(ast.literal_eval(cicE_str))

    @staticmethod
    def from_json(json_rule: dict, id_ident: str = "id") -> "cicE":

        """
            Converts a cicE on json into a interal cicE data type.

            Example:
                Input:
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

                Output:
                    "(['x','y'],['a','b']),'REQUIRES_ALL',('m','n','o')"
                    "(['x','y'],['a','b']),'REQUIRES_ALL',('m','n','o')"

            Return:
                cicE
        """
        return cicE([
            (tuple if json_rule['condition']['relation'] == "ALL" else list)(
                [
                    (tuple if sub_condition['relation'] == "ALL" else list)(
                        [
                            component[id_ident]
                            for component in sub_condition['components']
                        ]
                    )
                    for sub_condition in json_rule['condition']['subConditions']
                ]
            ),
            json_rule['consequence']['ruleType'],
            tuple(
                [x[id_ident] for x in json_rule['consequence']['components']]
            ),
            tuple(json_rule['consequence'].get("preferred", ()))
        ])

class cicEs(list):
    
    """
        Is a conjunction-list of `cicE` -items. 
    """
    pass