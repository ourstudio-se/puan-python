import copy
import typing
import functools
import itertools
import operator
import maz
import puan.logic.plog as pg

class application(dict):

    """
        application is a meta variant of a cic-rule where relations between variables
        are set based on variable properties. The variables in this context are called
        items and are objects which can hold any data but requires an "id".

        Methods
        -------
        to_cicJEs
            Converts an application and items to one or many configuration rules.
    """

    @staticmethod
    def _extract_value(from_dict: dict, selector_string: str) -> typing.Any:
        """
            Extracts value from a dictionary given a "selector string". A
            selector string seperates sub keys by a dot.

            E.g. from_dict = {'k': {'g': 1}}, selector_string = 'k.g'
            and value 1 is returned.

            Raises
            ------
                Exception
                    | If no keys in selector string.
                    | If key is selector string but key is not in dict.
                    | Key is not numeric nor one of '>' or '<'
                    | If any general error occurred when extracting value

            Returns
            -------
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
            return value if len(keys) == 0 else application._extract_value(value, ".".join(keys))
        elif isinstance(from_dict, list):
            if key.isnumeric():
                return application._extract_value(from_dict[int(key)], ".".join(keys))
            elif key in ["<", ">"]:
                return application._extract_value(from_dict[0] if key == "<" else from_dict[-1], ".".join(keys))
            else:
                raise Exception(f"when value is a list, key must be either a number or '<' or '>', got {key}")
        else:
            raise Exception(f"Cannot handle type {type(from_dict)}")

    @functools.lru_cache()
    def _operators_map():
        return {
            "==": operator.eq,
            "!=": operator.ne,
            "<":  operator.lt,
            "<=": operator.le,
            ">":  operator.gt,
            ">=": operator.ge,
        }

    @staticmethod
    def _validate_literal(literal: dict, item: dict) -> bool:
        """
            Validates if item[key] != value.

            Raises
            ------
                Exception
                    If no function exists for given operator.
                
                KeyError
                    If key not in item.

            Returns
            -------
                out : bool
        """
        operators = application._operators_map()
        if not literal["operator"] in operators:
            raise Exception(f"""Cannot handle operator '{literal["operator"]}'""")

        try:
            return operators.get(literal["operator"])(
                application._extract_value(item, literal["key"]),
                literal["value"],
            )
        except KeyError as ke:
            if not literal.get("skipIfKeyError", False):
                raise ke
        except Exception as e:
            raise Exception(f"Could not validate literal {literal} on item {item} because of error: {e}")

        return False

    @staticmethod
    def _validate_item_from(conjunction_selector: dict, item: dict) -> bool:

        """
            Validates an item with an operation list.

            Returns
            -------
                out : bool
        """

        return all(
            any(
                application._validate_literal(literal, item)
                for literal in disjunction_selector["literals"]
            )
            for disjunction_selector in conjunction_selector.get("disjunctions", [])
        )


    @staticmethod
    def _extract_items_from(conjunction_selector: dict, items: typing.List[dict]) -> iter:
        """
            Given an operation list and list of dictionaries, dictionaries are extracted
            based on the logic from operation_list.

            Returns
            -------
                out : list

            Notes
            -----
            If selector["active"] is true and no extractors are defined, then all items are returnd.
            If selector["active"] is false, then no item are returned.

        """
        if len(conjunction_selector.get("disjunctions", [])) == 0:
            iterator = (x for x in items)
        else:
            iterator = filter(
                functools.partial(
                    application._validate_item_from,
                    conjunction_selector,
                ),
                items,
            )

        return iterator

    @staticmethod
    def _apply_selector(selector: dict, to_items: typing.List[dict]) -> list:

        """
            Extracts items using selector's extractions and applies requirement checks.
            If any requirement is not fulfilled, then exception is raised.

            Raises
            ------
                Exception
                    | If operator not valid.
                    | If requirement is not fulfilled.

            Returns
            -------
                out : list(list)
        """
        requirement_checks = {
            "EMPTY": lambda items: len(items) == 0,
            "EXACTLY_ONE": lambda items: len(items) == 1,
            "NOT_EMPTY": lambda items: len(items) > 0,
        }

        extracted_items = list(application._extract_items_from(selector.get("conjunctionSelector", {}), to_items)) if selector["active"] else []
        for requirement in selector.get("requirements", []):
            if not requirement in requirement_checks:
                raise Exception(f"Operator '{requirement}' not a valid requirement operator")

            if not requirement_checks.get(requirement)(extracted_items):
                raise Exception(f"Requirement '{requirement.name}' not fulfilled on {selector.to_dict()}")

        return extracted_items

    @staticmethod
    def _group_by(items: list, by: typing.List[str], skip_item_if_fail: bool = False) -> list:

        """
            Group items by their keys `group_bys` -value's.

            Raises
            ------
                Exception
                    If `skip_item_if_fail` is False and failing to generate groups for any item.

            Returns
            -------
                out : List[Dict[str, ]]
        """

        grouping = {}
        for item in items:
            try:
                item_key_values = tuple([(gb, application._extract_value(item, gb)) for gb in by])
            except Exception as e:
                if skip_item_if_fail:
                    continue
                raise e

            if not item_key_values in grouping:
                grouping[item_key_values] = []
            grouping[item_key_values].append(item)

        return grouping

    @staticmethod
    def _apply_collector(collector: dict, to_items: typing.List[dict]) -> list:

        """
            Extracts items using collector and applies grouping.

            Returns
            -------
                out : list(list)
        """
        on_key = collector.get("groupBy", {}).get("onKey", "")
        selected_items = application._apply_selector(collector["selector"], to_items)
        if on_key == "":
            return [selected_items]
        else:
            # Remove keys from grouping
            return list(application._group_by(selected_items, [on_key]).values())

    def _explode_from_variables(self: dict) -> typing.List[dict]:
        """
            Replaces variable value from variables in application into
            many applications.

            Returns
            -------
                out : List (Application)
        """
        if len(self.get("variables", [])) == 0:
            return [self]
        else:
            return list(
                map(
                    lambda variable: application._replace_variables(self, variable),
                    self.get("variables", []),
                )
            )


    def _replace_variables(self: dict, variable: dict, variable_sign: str = "$") -> "application":

        """
            Replaces literal values with variable if match. Returns a copy of application.

            Returns
            -------
                out : Application
        """

        _application = copy.deepcopy(self)
        for collector in [_application["source"], _application["target"]]:
            for disjunction in collector["selector"].get("conjunctionSelector", {}).get("disjunctions", []):
                for literal in disjunction["literals"]:
                    if literal["value"].startswith(variable_sign) and literal["value"][1:] == variable["key"]:
                        literal["value"] = variable["value"]

        return _application

    def to_plog(self: dict, from_items: typing.List[dict], id_key: str = "id", model_id: str = None) -> pg.AtLeast:

        """
        Converts an application and items to plog.AtLeast logic system.

        Returns
        -------
            out : puan.plog.AtLeast

        Examples
        --------

        Here we want to set that each category item (such as bottoms, shoes and jeans) requires exactly one
        of the items that has category.id as "bottoms", "shoes" or "jeans". In result, we'll return three cicJE instances
        where one binds the item "bottom" to requires exclusively all items with category.id == "bottoms", another the item "shoes"
        to all items with category.id == "shoes" and lastly one binding the item "jeans" to all items with category.id == "jeans".

            >>> import puan.logic.sta as sta
            >>> a = sta.application({
            ...     "variables": [
            ...         {
            ...             "key": "variable",
            ...             "value": "bottoms"
            ...         },
            ...         {
            ...             "key": "variable",
            ...             "value": "shoes"
            ...         },
            ...         {
            ...             "key": "variable",
            ...             "value": "jeans"
            ...         }
            ...     ],
            ...     "source": {
            ...         "selector": {
            ...             "active": True,
            ...             "conjunctionSelector": {
            ...                 "disjunctions": [
            ...                     {
            ...                         "literals": [
            ...                             {
            ...                                 "key": "id",
            ...                                 "operator": "==",
            ...                                 "value": "$variable"
            ...                             }
            ...                         ]
            ...                     }
            ...                 ]
            ...             }
            ...         }
            ...     },
            ...     "target": {
            ...         "selector": {
            ...             "active": True,
            ...             "conjunctionSelector": {
            ...                 "disjunctions": [
            ...                     {
            ...                         "literals": [
            ...                             {
            ...                                 "key": "category.id",
            ...                                 "operator": "==",
            ...                                 "value": "$variable"
            ...                             }
            ...                         ]
            ...                     }
            ...                 ]
            ...             }
            ...         }
            ...     },
            ...     "apply": {
            ...         "ruleType": "REQUIRES_EXCLUSIVELY"
            ...     }
            ... })
            >>> a.to_plog([
            ...     {
            ...         "id": "4c2f9300-cc0e-42c6-b5c8-75ec5bcf4532",
            ...         "name": "Loose jeans",
            ...         "category": {
            ...             "name": "Jeans",
            ...             "id": "jeans"
            ...         }
            ...     },
            ...     {
            ...         "id": "83893701-473c-44e9-9881-a9a403a8a0fc",
            ...         "name": "Regular Mid Wash jeans",
            ...         "category": {
            ...             "name": "Jeans",
            ...             "id": "jeans"
            ...         }
            ...     },
            ...     {
            ...         "id": "1dcb5259-73db-4e73-a2d0-2b883715ee18",
            ...         "name": "Slim Stretch Chinos",
            ...         "category": {
            ...             "name": "Trousers",
            ...             "id": "trousers"
            ...         }
            ...     },
            ...     {
            ...         "id": "517e4b9d-697d-47b4-9701-965c7d46a927",
            ...         "name": "Regular Trousers Cotton Linen",
            ...         "category": {
            ...             "name": "Trousers",
            ...             "id": "trousers"
            ...         }
            ...     },
            ...     {
            ...         "id": "670b14b7-91e8-4045-8bdc-0e24d152c826",
            ...         "name": "T-shirt",
            ...         "category": {
            ...             "name": "T-shirts",
            ...             "id": "t-shirts"
            ...         }
            ...     },
            ...     {
            ...         "id": "1b32e500-3999-4a09-92d1-866f6970153f",
            ...         "name": "T-shirt",
            ...         "category": {
            ...             "name": "T-shirts",
            ...             "id": "t-shirts"
            ...         }
            ...     },
            ...     {
            ...         "id": "e2f97d5e-d4fe-4a8c-933f-43ddb0bd21e6",
            ...         "name": "Oxford Shirt",
            ...         "category": {
            ...             "name": "Shirts",
            ...             "id": "shirts"
            ...         }
            ...     },
            ...     {
            ...         "id": "8a3a0c21-f6a7-4447-bb6a-d278d7077aaa",
            ...         "name": "Relaxed Oxford Shirt",
            ...         "category": {
            ...             "name": "Shirts",
            ...             "id": "shirts"
            ...         }
            ...     },
            ...     {
            ...         "id": "00cba936-1b43-4422-bb1c-b8c9c5b0f173",
            ...         "name": "Relaxed Cotton Twill Overshirt",
            ...         "category": {
            ...             "name": "Shirts",
            ...             "id": "shirts"
            ...         }
            ...     },
            ...     {
            ...         "id": "02ece6a3-a5bd-4ff9-9256-26b0938a621e",
            ...         "name": "Heavy Knit Wool Jumper",
            ...         "category": {
            ...             "name": "Knits",
            ...             "id": "knits"
            ...         }
            ...     },
            ...     {
            ...         "id": "59079abb-8fae-402a-9e44-126165a95fd7",
            ...         "name": "Relaxed Heavyweight Hoodie",
            ...         "category": {
            ...             "name": "sweaters",
            ...             "id": "sweaters"
            ...         }
            ...     },
            ...     {
            ...         "id": "14ec21ec-5892-45ae-adb1-c7dc12b11379",
            ...         "name": "French Terry Sweatshirt",
            ...         "category": {
            ...             "name": "sweaters",
            ...             "id": "sweaters"
            ...         }
            ...     },
            ...     {
            ...         "id": "71a02a66-2614-470d-afd1-c858470e1107",
            ...         "name": "New Balance 997H",
            ...         "category": {
            ...             "name": "Sneakers",
            ...             "id": "sneakers"
            ...         }
            ...     },
            ...     {
            ...         "id": "5c462102-f15d-4cbd-872e-a2a9df5446d5",
            ...         "name": "Saucony Azura Trainers",
            ...         "category": {
            ...             "name": "Sneakers",
            ...             "id": "sneakers"
            ...         }
            ...     },
            ...     {
            ...         "id": "4b89a145-9c2e-479d-8ff3-e8f96b31cc6a",
            ...         "name": "Veja Esplar Trainers",
            ...         "category": {
            ...             "name": "Sneakers",
            ...             "id": "sneakers"
            ...         }
            ...     },
            ...     {
            ...         "id": "ffed933f-8036-43db-89e4-569423840dd8",
            ...         "name": "Leather Chelsea Boots",
            ...         "category": {
            ...             "name": "Boots",
            ...             "id": "boots"
            ...         }
            ...     },
            ...     {
            ...         "name": "Bottoms",
            ...         "id": "bottoms",
            ...         "category": {
            ...             "name": "Clothing",
            ...             "id": "clothing"
            ...         }
            ...     },
            ...     {
            ...         "name": "Tops",
            ...         "id": "tops",
            ...         "category": {
            ...             "name": "Clothing",
            ...             "id": "clothing"
            ...         }
            ...     },
            ...     {
            ...         "name": "Shoes",
            ...         "id": "shoes",
            ...         "category": {
            ...             "name": "Clothing",
            ...             "id": "clothing"
            ...         }
            ...     },
            ...     {
            ...         "name": "Jeans",
            ...         "id": "jeans",
            ...         "category": {
            ...             "name": "Bottoms",
            ...             "id": "bottoms"
            ...         }
            ...     },
            ...     {
            ...         "name": "Trousers",
            ...         "id": "trousers",
            ...         "category": {
            ...             "name": "Bottoms",
            ...             "id": "bottoms"
            ...         }
            ...     },
            ...     {
            ...         "name": "Shorts",
            ...         "id": "shorts",
            ...         "category": {
            ...             "name": "Bottoms",
            ...             "id": "bottoms"
            ...         }
            ...     },
            ...     {
            ...         "name": "T-shirts",
            ...         "id": "t-shirts",
            ...         "category": {
            ...             "name": "Tops",
            ...             "id": "tops"
            ...         }
            ...     },
            ...     {
            ...         "name": "Sweaters",
            ...         "id": "sweaters",
            ...         "category": {
            ...             "name": "Tops",
            ...             "id": "tops"
            ...         }
            ...     },
            ...     {
            ...         "name": "Knits",
            ...         "id": "knits",
            ...         "category": {
            ...             "name": "Tops",
            ...             "id": "tops"
            ...         }
            ...     },
            ...     {
            ...         "name": "Shirts",
            ...         "id": "shirts",
            ...         "category": {
            ...             "name": "Tops",
            ...             "id": "tops"
            ...         }
            ...     },
            ...     {
            ...         "name": "Sneakers",
            ...         "id": "sneakers",
            ...         "category": {
            ...             "name": "Shoes",
            ...             "id": "shoes"
            ...         }
            ...     },
            ...     {
            ...         "name": "Boots",
            ...         "id": "boots",
            ...         "category": {
            ...             "name": "Shoes",
            ...             "id": "shoes"
            ...         }
            ...     }
            ... ])
            VARea3404078e1fc1538b20464e99e81ff51bd0d7773686ffcb1cd7c6a560ba8ae4: +(VAR0aadd9ea54a7ca1f1e04b4cb522e761705c6fd9f105ec16658b8007c977d2470,VAR639a560d52db0adacb1ab70a97892ea14387ecea7c05383c15c778e6e00a7606,VARdc403dd989722bd60cbe3a3d22f7413577e85fa9a9276dd423c28bafab562e8a)>=3
    """
        n_ge_one = functools.partial(filter, maz.compose(maz.pospartial(operator.ge, [(1,1)]), len))
        return pg.All(
            *itertools.chain(
                *map(
                    lambda _application: list(
                        itertools.starmap(
                            lambda source_items, target_items: pg.Imply.from_cicJE({
                                "condition": {
                                    "relation": "ALL",
                                    "subConditions": [
                                        {
                                            "relation": _application["apply"].get("conditionRelation", "ALL"),
                                            "components": [
                                                {
                                                    "id": str(application._extract_value(source_item, id_key)),
                                                    "bounds": source_item.get("bounds", (0,1))
                                                }
                                                for source_item in source_items
                                            ]
                                        }
                                    ]
                                } if len(source_items) > 0 else {},
                                "consequence": {
                                    "ruleType":_application["apply"]["ruleType"],
                                    "components": [
                                        {
                                            "id": str(application._extract_value(target_item, id_key)),
                                            "bounds": target_item.get("bounds", (0,1))
                                        }
                                        for target_item in target_items
                                    ]
                                }
                            }),
                            itertools.product(
                                application._apply_collector(_application["source"], from_items), 
                                n_ge_one(
                                    application._apply_collector(_application["target"], from_items)
                                )
                            )
                        )
                    ),
                    application._explode_from_variables(self)
                )
            ),
            variable=model_id,
        )

    @staticmethod
    def to_all_proposition(applications: typing.List["application"], from_items: typing.List[dict], id_key: str = "id", id: str = None) -> pg.All:

        """
            Compiles into a `pg.All` proposition from a list of applications and a list of items.

            Returns
            -------
                out : pg.All
        """

        return pg.All(
            *itertools.chain(
                *map(
                    lambda x: x.to_plog(from_items).propositions,
                    map(
                        application,
                        applications
                    )
                )
            ),
            variable=id
        )