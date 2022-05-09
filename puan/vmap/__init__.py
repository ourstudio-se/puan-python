import copy
import numpy

class value_map(dict):

    """
        `value_map` is a dict and a way of storing matrix-data more compressed.
        Each key

        Methods
        -------
        to_matrix
            Expands the value map into a numpy matrix.
        merge
            Recursively merges value maps into one value map. (static)

    """

    def to_matrix(self: dict, m_max: int = 0, dtype = numpy.int16) -> numpy.ndarray:

        """
        Expands the value map into a numpy matrix.

        Parameters
        ----------
        m_max : int
            Maximimum column width
        d_type : data-type, optional
            The desired data-type for the array, e.g., `numpy.int8`.  Default is `numpy.int16`

        Examples
        --------
        >>> value_map({1: [[0, 0, 2, 1], [0, 4, 2, 1]],
        >>>            2: [[1, 1, 2], [3, 4, 0]]}).to_matrix()
        array([[1, 0, 0, 0, 1],
               [0, 1, 0, 2, 2],
               [2, 0, 1, 0, 0]], dtype=int16)


        Returns
        -------
        numpy.ndarray: (n x m)
        """
        if self == {}:
            # A linprog system has at least the support vector
            return numpy.zeros((1, m_max+1))

        n,m = 0,m_max
        for values in self.values():
            _n, _m = numpy.max(values, axis=1) if (len(values[0]) > 0 and len(values[1]) > 0) else (0, 0)

            n = _n if _n > n else n
            m = _m if _m > m else m

        M = numpy.zeros((n+1, m+1), dtype=dtype)
        for value, indices in self.items():
            M[tuple(indices)] = value

        return M

    @staticmethod
    def merge(*value_maps) -> dict:
        """
        Recursively merges value maps into one value map, such that value mapps to the
        right are appended to the first value map of the iterable.

        Parameters
        ----------
        value_maps : iterable (value_map)

        Returns
        -------
        dict (value map): value -> [[row_idxs], [col_idxs]]

        Examples
        --------
        >>> v1 = value_map({1: [[0, 1, 3], [0, 1, 3]],
        >>>               -1: [[1, 2], [0, 2]]})
        >>> v2 = value_map({1: [[0, 1], [1, 4]],
        >>>                2: [[1, 2], [0, 2]]})
        >>> v3 = value_map({3: [[1, 1], [1, 4]]})
        >>> v1.merge([v1, v2, v3])
        """
        if len(value_maps) == 0:
            return value_map({})

        if len(value_maps) == 1:
            return value_map(value_maps[0])

        value_map_list = list(value_maps)
        value_map_left, value_map_right = value_map_list[0], value_map_list[1]
        if value_map_left == {} and value_map_right == {}:
            return value_map({})

        elif value_map_left == {}:
            return value_map(value_map_right)

        elif value_map_right == {}:
            return value_map(value_map_left)

        highest_idx = 0
        for _, (row_idxs, _) in value_map_left.items():
            for row_idx in row_idxs:
                if row_idx > highest_idx:
                    highest_idx = row_idx

        merged_value_map = copy.deepcopy(value_map_left)
        for value, (row_idxs, col_idxs) in value_map_right.items():
            merged_value_map.setdefault(value, [[], []])
            merged_value_map[value][0] += [highest_idx+row_idx+1 for row_idx in row_idxs]
            merged_value_map[value][1] += col_idxs

        return value_map.merge(merged_value_map, *value_map_list[2:])

"""
    Function binding to function variables
    that should directly be accessible through
    puan.vmap.* on first level (e.g puan.vmap.merge(...))
"""

merge = value_map.merge
to_matrix = value_map.to_matrix
