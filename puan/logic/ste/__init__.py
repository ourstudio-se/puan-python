import copy
import typing
import functools
import operator

def extract_value(from_dict: dict, selector_string: str) -> typing.Any:
    """
        Extracts value from a dictionary given a "selector string". A
        selector string seperates sub keys by a dot.

        E.g. from_dict = {'k': {'g': 1}}, selector_string = 'k.g'
        and value 1 is returned.

        Return:
            any
    """
    keys = selector_string.split(".")
    if len(keys) == 0:
        raise Exception("No keys in selector string")

    key = keys.pop(0)
    if isinstance(from_dict, dict):
        if not key in from_dict:
            raise KeyError(f"No key '{key}' in dictionary {from_dict}")

        value = from_dict.get(key)
        return value if len(keys) == 0 else extract_value(value, ".".join(keys))
    elif isinstance(from_dict, list):
        if key.isnumeric():
            return extract_value(from_dict[int(key)], ".".join(keys))
        elif key in ["<", ">"]:
            return extract_value(from_dict[0] if key == "<" else from_dict[-1], ".".join(keys))
        else:
            raise Exception(f"when value is a list, key must be either a number or '<' or '>', got {key}")
    else:
        raise Exception(f"Cannot handle type {type(from_dict)}")

@functools.lru_cache()
def operators_map():
    return {
        "==": operator.eq,
        "!=": operator.ne,
        "<":  operator.lt,
        "<=": operator.le,
        ">":  operator.gt,
        ">=": operator.ge,
    }

def validate_literal(literal: dict, item: dict) -> bool:
    """
        Validates if item[key] != value.

        Return:
            bool
    """
    operators = operators_map()
    if not literal["operator"] in operators:
        raise Exception(f"""Cannot handle operator '{literal["operator"]}'""")

    try:
        return operators.get(literal["operator"])(
            extract_value(item, literal["key"]),
            literal["value"],
        )
    except KeyError as ke:
        if not literal.get("skipIfKeyError", False):
            raise ke
    except Exception as e:
        raise Exception(f"Could not validate literal {literal} on item {item} because of error: {e}")

    return False

def validate_item_from(conjunction_selector: dict, item: dict) -> bool:

    """
        Validates an item with an operation list.

        Return:
            bool
    """

    return all(
        any(
            validate_literal(literal, item)
            for literal in disjunction_selector["literals"]
        )
        for disjunction_selector in conjunction_selector.get("disjunctions", [])
    )


def extract_items_from(conjunction_selector: dict, items: typing.List[dict]) -> iter:
    """
        Given an operation list and list of dictionaries, dictionaries are extracted
        based on the logic from operation_list.

        NOTE: If selector["active"] is true and no extractors are defined, then all items are returnd.
              If selector["active"] is false, then no item are returned.

        Return:
            list
    """
    if len(conjunction_selector.get("disjunctions", [])) == 0:
        iterator = (x for x in items)
    else:
        iterator = filter(
            functools.partial(
                validate_item_from,
                conjunction_selector,
            ),
            items,
        )

    return iterator

def apply_selector(selector: dict, to_items: typing.List[dict]) -> list:

    """
        Extracts items using selector's extractions and applies requirement checks.
        If any requirement is not fulfilled, then exception is raised.

        Return:
            list(list)
    """
    requirement_checks = {
        "EMPTY": lambda items: len(items) == 0,
        "EXACTLY_ONE": lambda items: len(items) == 1,
        "NOT_EMPTY": lambda items: len(items) > 0,
    }

    extracted_items = list(extract_items_from(selector.get("conjunctionSelector", {}), to_items)) if selector["active"] else []
    for requirement in selector.get("requirements", []):
        if not requirement in requirement_checks:
            raise Exception(f"Operator '{requirement}' not a valid requirement operator")

        if not requirement_checks.get(requirement)(extracted_items):
            raise Exception(f"Requirement '{requirement.name}' not fulfilled on {selector.to_dict()}")

    return extracted_items

def group_by(items: list, by: typing.List[str], skip_item_if_fail: bool = False) -> list:

    """
        Group items by their keys `group_bys` -value's.

        Return:
            List[Dict[str, ]]
    """

    grouping = {}
    for item in items:
        try:
            item_key_values = tuple([(gb, extract_value(item, gb)) for gb in by])
        except Exception as e:
            if skip_item_if_fail:
                continue
            raise e

        if not item_key_values in grouping:
            grouping[item_key_values] = []
        grouping[item_key_values].append(item)

    return grouping

def apply_collector(collector: dict, to_items: typing.List[dict]) -> list:

    """
        Extracts items using collector and applies grouping.

        Return:
            list(list)
    """
    on_key = collector.get("groupBy", {}).get("onKey", "")
    selected_items = apply_selector(collector["selector"], to_items)
    if on_key == "":
        return [selected_items]
    else:
        # Remove keys from grouping
        return list(group_by(selected_items, [on_key]).values())

def application_from_variable(application: dict, variable: dict, variable_sign: str = "$") -> dict:

    """
        Replaces literal values with variable if match. Returns a copy of application.

        Return:
            Application
    """

    _application = copy.deepcopy(application)
    for collector in [_application["source"], _application["target"]]:
        for disjunction in collector["selector"].get("conjunctionSelector", {}).get("disjunctions", []):
            for literal in disjunction["literals"]:
                if literal["value"].startswith(variable_sign) and literal["value"][1:] == variable["key"]:
                    literal["value"] = variable["value"]

    return _application

def explode_from_variables(application: dict) -> typing.List[dict]:
    """
        Replaces variable value from variables in application into
        many applications.

        Return:
            List (Application)
    """
    if len(application.get("variables", [])) == 0:
        return [application]
    else:
        return list(
            map(
                lambda variable: application_from_variable(application, variable),
                application.get("variables", []),
            )
        )


def application2rules(application: dict, from_items: typing.List[dict], id_key: str = "id") -> iter:

    """
        Converts an application and items to one or many configuration rules.

        Return:
            iterator (ConfigRule)
    """
    for _application in explode_from_variables(application):

        source_items_group = apply_collector(_application["source"], from_items)
        target_items_group = apply_collector(_application["target"], from_items)

        for source_items in source_items_group:
            for target_items in target_items_group:

                if not (source_items or target_items):
                    continue

                yield {
                    "condition": {
                        "relation": "ALL",
                        "subConditions": [
                            {
                                "relation": _application["apply"].get("conditionRelation", "ALL"),
                                "components": [
                                    {"id": extract_value(source_item, id_key)}
                                    for source_item in source_items
                                ]
                            }
                        ]
                    } if len(source_items) > 0 else {},
                    "consequence": {
                        "ruleType":_application["apply"]["ruleType"],
                        "components": [
                            {"id": extract_value(target_item, id_key)}
                            for target_item in target_items
                        ]
                    }
                }

def components_from(from_items: typing.List[dict], selector: dict = None, id_key: str = None, ids: typing.List[str] = []) -> list:

    """
        Generates components from either a selector or a list of ids.

        Return:
            list (Component)
    """

    if (not selector["active"] and len(ids) == 0) or (selector["active"] and len(ids) > 0):
        raise Exception(f"Either a selector is given *xor* a list of id's. Not both nor none. {selector}")

    if selector["active"]:
        components = [
            {"id": extract_value(item, id_key)}
            for item in apply_selector(selector, from_items)
        ]
    else:
        components = [{"id": id} for id in ids]

    return components

def specification2rule(specification: dict, from_items: typing.List[dict], id_key: str = "id") -> dict:

    """
        Converts a specification and items to a configuration rule.

        Return:
            ConfigRule
    """
    return {
        "id":specification.get("id", ""),
        "condition": {
            "relation":specification["condition"]["relation"],
            "subConditions":[
                {
                    "relation":sub_condition["relation"],
                    "components":components_from(
                        from_items=from_items,
                        selector=sub_condition.get("selector", {"active": False}),
                        id_key=id_key,
                        ids=sub_condition.get("ids", [])
                    )
                }
                for sub_condition in specification["condition"]["subConditions"]
            ]
        },
        "consequence": {
            "ruleType": specification["consequence"]["ruleType"],
            "components": components_from(
                from_items=from_items,
                selector=specification["consequence"]["selector"],
                id_key=id_key,
                ids=specification["consequence"].get("ids", []),
            )
        }
    }

def settings2rules(settings: dict, from_items: typing.List[dict]) -> iter:

    """
        Converts an Application Set and items into list of configuration rules.

        Return:
            iterator (ConfigRule)
    """
    return itertools.chain(
        [
            specification2rule(specification, from_items)
            for specification in settings.get("specifications", [])
        ],
        *(
            application2rules(application, from_items)
            for application in settings.get("applications", [])
        ),
    )

def validate_rule(rule: dict) -> bool:

    """
        Return true if rule is valid, false if not.

        Return:
            bool
    """
    if rule["condition"] is not None:
        for sub_condition in rule["condition"]["subConditions"]:
            if len(sub_condition["components"]) == 0:
                return False

    if len(rule["consequence"]["components"]) == 0:
        return False

    return True