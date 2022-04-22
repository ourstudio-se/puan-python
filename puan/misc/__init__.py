import typing

def or_get(d: dict, keys: list, default_value = None) -> typing.Any:

    """
        # or_get
        is useful when you are not sure exactly
        how the keys looks like. You pass a list
        of candidate keys. The first that matches
        will be returned. If no match, then a
        KeyError exception is raised.

        :param d: dictionary
        :param keys: list of candidate keys
        :default value: value that is returned if no key matches (and default is not None)

        :return: value of d[k]
    """

    for k in keys:
        if k in d:
            return d[k]

    if default_value is not None:
        return default_value

    raise KeyError(keys)

def or_replace(d: dict, keys: list, value: typing.Any):
    """
        # or_replace
        will replace the first key value in keys
        that exists, with `value`. If no keys exists,
        KeyError is raised.

        :param d: dictionary
        :param keys: list of candidate keys
        :value: value that is set

        :return: value of d[k]
    """
    for k in keys:
        if k in d:
            d[k] = value
            return d

    raise KeyError(keys)