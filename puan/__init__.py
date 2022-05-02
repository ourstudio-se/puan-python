import numpy
import typing
import functools

class ge_polytope(numpy.ndarray):
    """
        A numpy.ndarray sub class and a system of linear inequalities forming
        a high dimensional polytope. The "ge" stands for "greater or equal" (:math:`\\ge`)
        which represents the relation between :math:`A` and :math:`b` (as in :math:`Ax \\ge b`).

        Attributes
        ----------
        See numpy.array

        Methods
        -------
        to_value_map
            Reduces the polytope into a value map.
        to_linalg
            assumes support vector index 0 in polytope and returns :math:`A, b` as in :math:`Ax \\ge b`
        reducable_columns_approx
            Returns what columns are reducable under approximate condition.
        reduce_columns
            Reducing columns from polytope from columns_vector.
        reducable_rows
            Returns a boolean vector indicating what rows are reducable.
        reduce_rows
            Reduces rows from a rows_vector where num of rows of M equals
            size of rows_vector.
        reducable_rows_and_columns
            Returns reducable rows and columns of given polytope.
        reduce
            Reduces matrix polytope by information passed in rows_vector and columns_vector.
        neglectable_columns
            Returns neglectable columns of given polytope `ge_polytope`.
        neglect_columns
            Neglects columns from a columns_vector
        isin
            Checks if points are inside the polytope.

    """

    def __new__(cls, input_array):
        return numpy.asarray(input_array).view(cls)

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def to_value_map(self: numpy.ndarray) -> dict:

        """
            Reduces the polytope into a value map.

            Returns
            -------
                out : dictionary: value -> indices

            Notes
            -----
                Since zeros are excluded in a value map, leading zero rows/columns will not be included.

            Examples
            --------
                >>> ge_polytope = ge_polytope(numpy.array([
                >>>     [0,-1, 1, 0, 0],
                >>>     [0, 0,-1, 1, 0],
                >>>     [0, 0, 0,-1, 1],
                >>> ]))
                >>> ge_polytope.to_value_map()
                {1: [[0, 1, 2], [2, 3, 4]], -1: [[0, 1, 2], [1, 2, 3]]}

        """
        return dict(map(lambda v:
                    (v, # Dict key
                    list(map(lambda x: x[1], enumerate(numpy.argwhere(self == v).T.tolist())))), # Dict value
                    set(self[self != 0])))

    def to_linalg(self: numpy.ndarray) -> tuple:
        """
            Assumes support vector index 0 in polytope
            and returns :math:`A, b` as in :math:`Ax \\ge b`

            Returns
            -------
                out : tuple
                    out[0] : A\n
                    out[1] : b

            Examples
            --------
                >>> ge = ge_polytope(np.array([
                >>>    [0,-1, 1, 0, 0],
                >>>    [0, 0,-1, 1, 0],
                >>>    [0, 0, 0,-1, 1],
                >>> ]))
                >>> ge.to_linalg()
                (ge_polytope([
                    [-1, 1, 0, 0],
                    [0, -1, 1, 0],
                    [0, 0, -1, 1]]),
                ge_polytope([0,0,0]))
        """
        if self.ndim < 2:
            A = self[1:].copy()
            b = self[0].copy()
        else:
            A = self[:, 1:].copy()
            b = self.T[0].copy()
        return A, b

    def reducable_columns_approx(self: numpy.ndarray) -> numpy.ndarray:
        """
            Returns what columns are reducable under approximate condition.
            Columns with positive values could be assumed.
            Columns with negative values could be removed (not-assumed).

            Returns
            -------
                out : numpy.ndarray (vector)

            See also
            --------
                reduce : Reduces matrix polytope by information passed in rows_vector and columns_vector.
                reduce_columns : Reducing columns from polytope from columns_vector where a positive number meaning "assume" and a negative number meaning "not assume".
                reducable_rows_and_columns : Returns reducable rows and columns of given polytope.
                reduce_rows : Reduces rows from a rows_vector where num of rows of M equals size of rows_vector.
                reducable_rows : Returns a boolean vector indicating what rows are reducable.

        """
        A, b = ge_polytope.to_linalg(self)
        r = (A*((A*(A <= 0) + (A*(A > 0)).sum(axis=1).reshape(-1,1)) < b.reshape(-1,1))) + A*((A * (A > 0)).sum(axis=1) == b).reshape(-1,1)
        return r.sum(axis=0)

    def reduce_columns(self: numpy.ndarray, columns_vector: numpy.ndarray) -> numpy.ndarray:

        """
            Reducing columns from polytope from columns_vector where a positive number meaning "assume"
            and a negative number meaning "not assume".

            Parameters
            ----------
            columns_vector : ndarray
                The polytope is reduced column-wise by equally many positives and negatives in columns-vector.

            Returns
            -------
                out : numpy.ndarray

            See also
            --------
                reduce : Reduces matrix polytope by information passed in rows_vector and columns_vector.
                reducable_rows_and_columns : Returns reducable rows and columns of given polytope.
                reduce_rows : Reduces rows from a rows_vector where num of rows of M equals size of rows_vector.
                reducable_rows : Returns a boolean vector indicating what rows are reducable.
                reducable_columns_approx : Returns what columns are reducable under approximate condition.

            Notes
            -----
                :math:`M` is concatenated :math:`A, b` (as in :math:`Ax \\ge b`), where :math:`b == A.T[0]`

            Examples
            --------
                >>> ge_polytope = ge_polytope(numpy.array([
                >>>     [0,-1, 1, 0, 0],
                >>>     [0, 0,-1, 1, 0],
                >>>     [0, 0, 0,-1, 1],
                >>> ]))
                >>> columns_vector = numpy.ndarray([1, 0,-1, 0]) # meaning assume index 0
                >>> ge_polytope.reduce_columns(columns_vector)
                numpy.ndarray([
                        [1, 1, 0],
                        [0,-1, 0],
                        [0, 0, 1],
                    ])

        """

        A, b = ge_polytope.to_linalg(self)
        _b = b - (A.T*(columns_vector > 0).reshape(-1,1)).sum(axis=0)
        _A = numpy.delete(A, numpy.argwhere(columns_vector != 0).T[0], 1)
        return ge_polytope(numpy.append(_b.reshape(-1,1), _A, axis=1))

    def reducable_rows(self: numpy.ndarray) -> numpy.ndarray:
        """
            Returns a boolean vector indicating what rows are reducable.

            Returns
            -------
                out : numpy.ndarray (vector)

            See also
            --------
                reduce : Reduces matrix polytope by information passed in rows_vector and columns_vector.
                reduce_columns : Reducing columns from polytope from columns_vector where a positive number meaning "assume" and a negative number meaning "not assume".
                reducable_rows_and_columns : Returns reducable rows and columns of given polytope.
                reduce_rows : Reduces rows from a rows_vector where num of rows of M equals size of rows_vector.
                reducable_columns_approx : Returns what columns are reducable under approximate condition.
        """
        A, b = ge_polytope.to_linalg(self)
        return (((A * (A < 0)).sum(axis=1) >= b)) + ((A >= 0).all(axis=1) & (b<=0))

    def reduce_rows(self: numpy.ndarray, rows_vector: numpy.ndarray) -> numpy.ndarray:

        """
            Reduces rows from a rows_vector where num of rows of M equals
            size of rows_vector. Each row in rows_vector == 0 is kept.

            Parameters
            ----------
                rows_vector : numpy.ndarray

            Returns
            -------
                out : ge_polytope

            See also
            --------
                reduce : Reduces matrix polytope by information passed in rows_vector and columns_vector.
                reduce_columns : Reducing columns from polytope from columns_vector where a positive number meaning "assume" and a negative number meaning "not assume".
                reducable_rows_and_columns : Returns reducable rows and columns of given polytope.
                reducable_rows : Returns a boolean vector indicating what rows are reducable.
                reducable_columns_approx : Returns what columns are reducable under approximate condition.
        """

        return self[rows_vector == 0]

    def reducable_rows_and_columns(self: numpy.ndarray) -> tuple:

        """
            Returns reducable rows and columns of given polytope.

            Returns
            -------
                out : tuple
                    out[0] : a vector equal size as ge_polytope's row size where 1 represents a removed row and 0 represents a kept row\n
                    out[1] : a vector with equal size as ge_polytope's column size where a positive number represents requireds and a negative number represents forbids

            See also
            --------
                reduce : Reduces matrix polytope by information passed in rows_vector and columns_vector.
                reduce_rows : Reduces rows from a rows_vector where num of rows of M equals size of rows_vector.
                reducable_rows : Returns a boolean vector indicating what rows are reducable.
                reduce_columns :  Reducing columns from polytope from columns_vector where a positive number meaning "assume" and a negative number meaning "not assume".
                reducable_columns_approx : Returns what columns are reducable under approximate condition.

            Examples
            --------
                >>> ge_polytope = ge_polytope(np.array([
                >>>     [ 0,-1, 1, 0, 0, 0], # 1
                >>>     [ 0, 0,-1, 1, 0, 0], # 2
                >>>     [-1,-1, 0,-1, 0, 0], # 3 1+2+3 -> Force not variable 0, not identified under the approximate conditions
                >>>     [ 1, 0, 0, 0, 1, 0], # Force variable 3
                >>>     [ 0, 0, 0, 0, 0,-1], # Force not variable 4
                >>>     [ 0, 1, 1, 0, 1, 0], # Redundant rule
                >>>     [ 0, 1, 1, 0, 1,-1], # Redundant when variable 4 forced not
                >>> ]))
                >>> reducable_rows_and_columns(ge_polytope)
                (array([0, 0, 0, 1, 1, 1, 1]), array([ 0,  0,  0,  1, -2]))

        """

        _M = self.copy()
        red_cols = ge_polytope.reducable_columns_approx(_M)
        red_rows = ge_polytope.reducable_rows(_M) * 1
        full_cols = numpy.zeros(_M.shape[1]-1, dtype=int)
        full_rows = numpy.zeros(_M.shape[0], dtype=int)
        while red_cols.any() | red_rows.any():
            _M = ge_polytope.reduce_columns(_M, red_cols)
            full_cols[full_cols == 0] = red_cols

            red_rows = ge_polytope.reducable_rows(_M) * 1
            _M = ge_polytope.reduce_rows(_M, red_rows)
            full_rows[full_rows == 0] = red_rows

            red_cols = ge_polytope.reducable_columns_approx(_M)
            red_rows = ge_polytope.reducable_rows(_M) * 1

        return full_rows, full_cols

    def reduce(self: numpy.ndarray, rows_vector: numpy.ndarray=None, columns_vector: numpy.ndarray=None) -> numpy.ndarray:
        """
            Reduces matrix polytope by information passed in rows_vector and columns_vector.

            Parameters
            ----------
            rows_vector : numpy.ndarray (optional)
                A vector of 0's and 1's where rows matching index of value 1 are removed.

            columns_vector : numpy.ndarray (optional)
                A vector of positive and negative integers where the positive represents active selections and negative
                represents active "not" selections. The polytope is reduced under those assumptions.

            Returns
            -------
                out : ge_polytope
                    Reduced polytope

            See also
            --------
                reducable_rows_and_columns : Returns reducable rows and columns of given polytope.
                reduce_rows : Reduces rows from a rows_vector where num of rows of M equals size of rows_vector.
                reducable_rows : Returns a boolean vector indicating what rows are reducable.
                reduce_columns :  Reducing columns from polytope from columns_vector where a positive number meaning "assume" and a negative number meaning "not assume".
                reducable_columns_approx : Returns what columns are reducable under approximate condition.

            Examples
            --------
                >>> self = numpy.array([
                >>>     [ 0,-1, 1, 0, 0, 0, 0],
                >>>     [ 0, 0,-1, 1, 0, 0, 0],
                >>>     [-1, 0, 0,-1,-1, 0, 0],
                >>>     [ 1, 0, 0, 0, 0, 1, 1],
                >>> ])
                >>> columns_vector = numpy.array([1,0,0,0,0,0])
                >>> reduce(self, columns_vector)
                numpy.array([
                        [ 1, 1, 0, 0, 0, 0],
                        [ 0,-1, 1, 0, 0, 0],
                        [-1, 0,-1,-1, 0, 0],
                        [ 1, 0, 0, 0, 1, 1],
                    ])
        """
        gp = self.copy()
        if rows_vector is not None:
            gp = ge_polytope.reduce_rows(gp, rows_vector)
        if columns_vector is not None:
            gp = ge_polytope.reduce_columns(gp, columns_vector)
        return gp

    def neglectable_columns(self: numpy.ndarray, patterns: numpy.ndarray) -> numpy.ndarray:
        """
            Returns neglectable columns of given polytope `ge_polytope` based on given patterns.
            Neglectable columns are the columns which doesn't differentiate the patterns in `ge_polytope`
            and the patterns not in `ge_polytope`

            Parameters
            ----------
                patterns : numpy.ndarray
                    known patterns of variables in the polytope.

            Returns
            -------
                out : numpy.ndarray
                    A 1-d ndarray of length equal to the number of columns of the ge_polytope,
                    with ones at corresponding columns which can be neglected, zero otherwise.

            See also
            --------
                neglect_columns : Neglects columns from a columns_vector where num of cols of :math:`M - 1` equals size of cols_vector.

            Notes
            -----
                This method is differentiating the patterns which are in ge_polytope from those that are not.
                Variables which are not in the patterns and has a positive number for any row in ge_polytope
                is considered non-neglectable.

            Examples
            --------
                >>> ge_polytope = numpy.array([
                >>>     [-1,-1,-1, 0, 0, 0, 1],
                >>>     [-1,-1, 0,-1, 0, 0, 1],
                >>> ])
                >>> patterns = numpy.array([
                >>>     [1, 1, 0],
                >>>     [0, 1, 1],
                >>>     [1, 0, 1]
                >>> ])
                >>> neglectable_columns(ge_polytope, patterns)
                numpy.array([0, 1, 1, 1, 1, 0])

        """
        A, b = ge_polytope.to_linalg(self)
        # Extend patterns to be of same shape as A
        _patterns = patterns.copy()
        _patterns = numpy.pad(_patterns, ((0,0), (0, A.shape[1] - _patterns.shape[1])), 'constant')

        # We will never neglect columns that aren't part of any pattern
        columns_not_in_patterns = (_patterns==0).all(axis=0)
        non_neglectable_columns = columns_not_in_patterns
        _A = A.copy()
        _A[:, columns_not_in_patterns] = 0
        _A[numpy.nonzero(_A)] = 1

        # Find which patterns are not in A
        patterns_not_in_A = _patterns[~(_patterns[:, None] == _A).all(-1).any(-1)]
        if patterns_not_in_A.shape[0]==0:
            # Possible to neglect everything except non neglectable columns
            return (~non_neglectable_columns).astype(int)

        # Find common pattern in A
        common_pattern_in_A = (_A == 1).all(axis=0)
        if not (patterns_not_in_A[:,common_pattern_in_A]==1).all(axis=1).any(axis=0):
            # Possible to neglect everything except the common pattern and the non neglectable columns
            return (~(common_pattern_in_A | non_neglectable_columns)).astype(int)
        return ((_A[:, (patterns_not_in_A==0).all(axis=0) & (non_neglectable_columns==0)]).any(axis=1).all()) * (patterns_not_in_A!=0).any(axis=0).astype(int)

    def neglect_columns(self: numpy.ndarray, columns_vector: numpy.ndarray) -> numpy.ndarray:
        """
            Neglects columns from a columns_vector where num of cols of :math:`M - 1` equals
            size of cols_vector. The entire column for col in columns_vector :math:`>0`
            is set to :math:`0` and the support vector is updated.

            Parameters
            ----------
                columns_vector : numpy.ndarray
                    A 1-d ndarray of length equal to the number of columns of :math:`A`.
                    Sets the entire column of :math:`A` to zero for corresponding entries of columns_vector which are :math:`>0`.

            Returns
            --------
                out : ge_polytope
                    ge_polytope with neglected columns set to zero and suupport vector updated accordingly.

            See also
            --------
                neglectable_columns : Returns neglectable columns of given polytope `ge_polytope` based on given patterns.

            Examples
            --------
                >>> ge_polytope = numpy.array([
                >>>     [0,-1, 1, 0, 0],
                >>>     [0, 0,-1, 1, 0],
                >>>     [0, 0, 0,-1, 1],
                >>>  ])
                >>> columns_vector = numpy.array([1, 0, 1, 0])
                >>> neglect_columns(ge_polytope, columms_vector)
                numpy.ndarray([
                        [ 1, 0, 1, 0, 0],
                        [-1, 0,-1, 0, 0],
                        [ 1, 0, 0, 0, 1],
                    ])


        """
        A, b = ge_polytope.to_linalg(self)
        _b = b - (A.T*(columns_vector > 0).reshape(-1,1)).sum(axis=0)
        _A = A
        _A[:, columns_vector>0] = 0
        return ge_polytope(numpy.append(_b.reshape(-1,1), _A, axis=1))

    def isin(self: numpy.ndarray, points: numpy.ndarray) -> numpy.ndarray:

        """
            Checks if points are inside the polytope.

            .. code-block::

                           / \\
                          /   \\
                point -> /  x  \\
                        /_ _ _ _\ <- polytope

            Parameters
            ----------
                points : numpy.ndarray
                    Points to be evaluated if inside of polytope.

            Returns
            -------
                out : numpy.ndarray.
                    Boolean vector indicating T if in polytope.

            Examples
            --------
                >>> ge = ge_polytope([[0,-2,1,1]])
                >>> ge.isin(numpy.array([
                >>>     [1,0,1],
                >>>     [1,1,1],
                >>>     [0,0,0]
                >>> ]))
                array([False, True, True])
        """
        if points.ndim > 2:
            return numpy.array(
                list(
                    map(self.isin, points)
                )
            )
        elif points.ndim == 2:
            A, b = ge_polytope.to_linalg(self)
            return numpy.array(
                (numpy.matmul(A, points.T) >= b.reshape(-1,1)).all(axis=0)
            )
        elif points.ndim == 1:
            return ge_polytope.isin(self, numpy.array([points]))[0]



class integer_ndarray(numpy.ndarray):
    """
        A numpy.ndarray sub class with only integers in it.

        Attributes
        ----------
        See numpy.array

        Methods
        -------
        truncate
            Takes an integer ndarray and truncates it into a vector.
        enfolds
            Checks if a linear inequality in the polytope enfolds *all* points in self.
        to_value_map
            reduces the matrix into a value map.
        from_list
            Turns a list (or list of list) of strings into an integer vector. (static)

    """

    def __new__(cls, input_array):
        return numpy.asarray(input_array, dtype=numpy.int64).view(cls)

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def truncate(self: numpy.ndarray) -> numpy.ndarray:

        """
            Takes an integer ndarray and truncates it into a vector. (This function is used in the context of
            combinatorial optimization where one assumes that items has positive and negative prios from user)

            Returns
            -------
                out : numpy.ndarray (1d)

            Notes
            -----
            A *prio vector* is a vector with negative and positive integer numbers where each number indicates a negative
            or positive prioritization. High number means higher prio. E.g. [ 1, 2, 3,-1,-2] means that index
            1 has higher prio than index 0, and index 4 has higher negative prio than index 3.

            A *prio matrix* consist of prio vectors but has also a prioritization between the vectors. So, a vector
            on row index 0 has lower prio than a row vector on index 1.

            Examples
            --------
                >>> self = integer_ndarray([
                >>>         [ 0, 0, 0, 0, 1, 2],
                >>>         [-1,-1,-1,-1, 0, 0],
                >>>         [ 0, 0, 1, 2, 0, 0],
                >>>         [ 0, 0, 1, 0, 0, 0]
                >>>     ])
                >>> self.truncate()
                numpy.array([-3,-3, 5, 4, 1, 2])

        """
        if self.ndim > 2:
            return integer_ndarray.truncate(
                integer_ndarray(
                    list(
                        map(
                            integer_ndarray.truncate,
                            self
                        )
                    )
                )
            )
        elif self.ndim == 2:
            # self_abs = numpy.abs(self)
            # neg_value_msk = self < 0
            # row_value_offset = numpy.pad(
            #         numpy.cumsum(
            #             self_abs.sum(axis=1)
            #         ),
            #         (1,0)
            #     )[:-1].reshape(-1,1) * (self != 0)
            # offset_abs = (self_abs + row_value_offset)
            # offset_ord = offset_abs * ~neg_value_msk + offset_abs * neg_value_msk * -1
            # offset_ord_rev = offset_ord[::-1]
            # min_non_zero_row_idx = (offset_ord_rev != 0).argmax(axis=0)
            # truncated = offset_ord_rev[min_non_zero_row_idx, numpy.arange(offset_abs.shape[1])]
            # truncated_neg_msk = truncated < 0
            # truncated_abs = numpy.abs(truncated)
            # non_zeros = numpy.nonzero(truncated_abs)
            # if non_zeros[0].size > 0:
            #     truncated_abs_norm = truncated_abs - (truncated_abs[non_zeros].min()-1)
            # else:
            #     truncated_abs_norm = truncated_abs
            # truncated_norm = truncated_abs_norm * ~truncated_neg_msk + truncated_abs_norm * truncated_neg_msk * -1
            # return truncated_norm
            return integer_ndarray.truncate(self.flatten()).reshape()
        elif self.ndim == 1:
            return self
        else:
            return -1

    def enfolds(self: numpy.ndarray, gp: ge_polytope) -> numpy.ndarray:

        """
            Checks if a linear inequality in the polytope enfolds *all* points in self.
            Two of three linear equalities enfolds both points here::


                           / \\
                          /   \  x <- point
                point -> /  x  \\
                        /_ _ _ _\ <- polytope

            Parameters
            ----------
                gp : ge_polytope

            Returns
            -------
                out : numpy.ndarray
                    boolean vector indicating T if linear inequality enfolds all points

            See also
            --------
                ge_polytope.isin : Checks if points are inside the polytope.

            Notes
            -----
                This function is the inverse of ge_polytope.isin

            Examples
            --------
                >>> ge = integer_ndarray([[1,1,1],[1,0,1]])
                >>> ge.enfolds([
                >>> [ 0,-1,-1, 0],
                >>> [ 0, 1, 0,-1]
                >>> ])
                array([False, True])

        """
        if self.ndim > 2:
            return numpy.array(
                list(
                    map(self.enfolds, self)
                )
            )
        elif self.ndim == 2:
            A, b = gp.to_linalg()
            return numpy.array(
                (numpy.matmul(A, self.T) >= b.reshape(-1,1)).all(axis=1)
            )
        elif self.ndim == 1:
            return integer_ndarray.enfolds(numpy.array([self]), gp)

    def to_value_map(self: numpy.ndarray, mapping: dict = {}) -> dict:

        """
            to_value_map reduces the matrix into a value map.

            Parameters
            ----------
                mapping : dict
                    An ordinary dict, mapping one value to another

            Returns
            -------
                dictionary: value -> indices

            Notes
            -----
                Since zeros are excluded in a value map, leading zero rows/columns will not be included.

        """
        return {
            value: [
                [
                    mapping[j] if j in mapping and i > 0 and j > 0 else j
                    for j in k
                ]
                for i, k in enumerate(numpy.argwhere(self == value).T.tolist())
            ]
            for value in set(self[self != 0])
        }

    @staticmethod
    def from_list(lst: list, context: typing.List[str]) -> "integer_ndarray":
        """
            Turns a list (or list of list) of strings into an integer vector, where each value represents
            which order the string in lst was positioned.

            Parameters
            ----------
            lst : list
                List of strings
            context : list
                List of strings

            Returns
            -------
                out : integer_ndarray

            Examples
            --------
                >>> variables   = ["a","c","b"]
                >>> context     = ["a","b","c","d"]
                >>> integer_ndarray.from_list(variables, context)
                integer_ndarray([1,3,2,0])

        """
        if len(lst) == 0:
            result = []
        elif isinstance(lst[0], list):
            result = list(map(functools.partial(integer_ndarray.from_list, context=context), lst))
        else:
            result = list(
                map(
                    lambda x: 1*(x in lst) and (1+lst.index(x)),
                    context
                )
            )

        return integer_ndarray(result)

class boolean_ndarray(integer_ndarray):
    """
        A numpy.ndarray sub class with only booleans in it.

        Attributes
        ----------
        See numpy.array

        Methods
        -------
        from_list
            Turns a list of strings into a boolean (0/1) vector. (static)
    """

    def __new__(cls, input_array):
        return numpy.asarray(input_array, dtype=numpy.int64).view(cls)

    def __array_finalize__(self, obj):
        if obj is None:
            return

    @staticmethod
    def from_list(lst: typing.List[str], context: typing.List[str]) -> "boolean_ndarray":
        """
            Turns a list of strings into a boolean (0/1) vector.

            Parameters
            ----------
            lst : list
                list of variables as strings

            context : list
                list of context variables as strings

            Returns
            -------
                out : list
                    booleans with same dimension as **context**

            Examples
            --------
                >>> variables = ["a","c","b"]
                >>> context = ["a","b","c","d"]
                >>> boolean_ndarray.from_list(variables, context)
                boolean_ndarray([1,1,1,0])

        """
        if len(lst) == 0:
            result = []
        elif isinstance(lst[0], list) or isinstance(lst[0], tuple):
            result = list(map(functools.partial(boolean_ndarray.from_list, context=context), lst))
        else:
            result = list(
                map(
                    lambda x: 1*(x in lst),
                    context
                )
            )

        return boolean_ndarray(result)

"""
    Function binding to function variables
    that should directly be accessible through
    puan.* on first level (e.g puan.to_linalg())
"""
to_value_map =                  ge_polytope.to_value_map
to_linalg =                     ge_polytope.to_linalg
reducable_columns_approx =      ge_polytope.reducable_columns_approx
reduce_columns =                ge_polytope.reduce_columns
reducable_rows =                ge_polytope.reducable_rows
reduce_rows =                   ge_polytope.reduce_rows
reducable_rows_and_columns =    ge_polytope.reducable_rows_and_columns
reduce =                        ge_polytope.reduce
neglectable_columns =           ge_polytope.neglectable_columns
neglect_columns =               ge_polytope.neglect_columns
isin =                          ge_polytope.isin
truncate =                      integer_ndarray.truncate
enfolds =                       integer_ndarray.enfolds
