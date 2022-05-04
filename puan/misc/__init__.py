import typing

def or_get(d: dict, keys: list, default_value = None) -> typing.Any:

    """
        Returns the value from dict based on first match of candidate keys.
        :code:`or_get` is useful when you are not sure exactly
        how the keys looks like. You pass a list
        of candidate keys. The first that matches
        will be returned. If no match, then a
        KeyError exception is raised.

        Parameters
        ----------
        d : dict

        keys : list of candidate keys

        default_value : value that is returned if no key matches (and default is not None)

        Returns
        -------
            out : value of d[k]

        See also
        --------
        or_replace : will replace the first key value in keys that exists, with :code:`value`.

        Examples
        --------
        >>> d = dict((("a", 1), ("b", 2)))
        >>> keys = ["1", "a"]
        >>> or_get(d, keys)
        1

        >>> d = dict((("a", 1), ("b", 2)))
        >>> keys = ["b", "a"]
        >>> or_get(d, keys)
        2

        >>> d = dict((("a", 1), ("b", 2)))
        >>> keys = [1]
        >>> default_value = 0
        >>> or_get(d, keys, default_value)
        0
    """

    for k in keys:
        if k in d:
            return d[k]

    if default_value is not None:
        return default_value

    raise KeyError(keys)

def or_replace(d: dict, keys: list, value: typing.Any):
    """
        :code:`or_replace` will replace the first key value in keys
        that exists, with :code:`value`. If no keys exists,
        KeyError is raised.

        Parameters
        ----------
        d : dict

        keys : list
            candidate keys

        value : value that is set

        Returns
        -------
            out : d
                Updated dict

        See also
        --------
        or_get : Returns the value from dict based on first match of candidate keys.

        Examples
        --------
        >>> d = dict((("a", 1), ("b", 2)))
        >>> keys = ["1", "a"]
        >>> value = "1"
        >>> or_replace(d, keys, value)
        {'a': '1', 'b': 2}

        >>> d = dict((("a", 1), ("b", 2)))
        >>> keys = ["b", "a"]
        >>> value = "1"
        >>> or_replace(d, keys, value)
        {'a': 1, 'b': '1'}
    """
    for k in keys:
        if k in d:
            d[k] = value
            return d

    raise KeyError(keys)