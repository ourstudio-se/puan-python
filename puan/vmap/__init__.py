import copy
import numpy

class value_map(dict):

    """
        `value_map` is a dict and a way of storing matrix-data more compressed.
        Each key
    """

    def to_matrix(self: dict, m_max: int = 0, dtype = numpy.int16) -> numpy.ndarray:

        """
            Expands the value map into a numpy matrix. 

            Parameters:
                m_max: maximimum column width

            Example: ...

            Return:
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
            `merge` merges recursively value maps into one value map.
            Since the row indices from one value map to the other may collide,
            each merge will first find the highest row index value in the left
            value map, and then add it onto all the row values of the right value map.

            Return:
                dict (value map): value -> [[row_idxs], [col_idxs]]
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