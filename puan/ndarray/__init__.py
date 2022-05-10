import numpy
import typing
import functools
import operator
import maz
import math
import puan
import puan.ndarray.npufunc as npufunc

class ge_polyhedron(numpy.ndarray):
    """
        A numpy.ndarray sub class and a system of linear inequalities forming
        a polyhedron. The "ge" stands for "greater or equal" (:math:`\\ge`)
        which represents the relation between :math:`A` and :math:`b` (as in :math:`Ax \\ge b`), i.e.
        polyhedron :math:`P=\{x \\in R^n \ |\  Ax \\ge b\}`.

        Attributes
        ----------
        See numpy.array

        Methods
        -------
        to_value_map
            Reduces the polyhedron into a value map.
        to_linalg
            assumes support vector index 0 in polyhedron and returns :math:`A, b` as in :math:`Ax \\ge b`
        reducable_columns_approx
            Returns what columns are reducable under approximate condition.
        reduce_columns
            Reducing columns from polyhedron from columns_vector.
        reducable_rows
            Returns a boolean vector indicating what rows are reducable.
        reduce_rows
            Reduces rows from a rows_vector where num of rows of M equals
            size of rows_vector.
        reducable_rows_and_columns
            Returns reducable rows and columns of given polyhedron.
        reduce
            Reduces matrix polyhedron by information passed in rows_vector and columns_vector.
        neglectable_columns
            Returns neglectable columns of given polyhedron `ge_polyhedron`.
        neglect_columns
            Neglects columns in :math:`A` from a columns_vector.
        separable
            Checks if points are inside the polyhedron.
        ineq_separate_points
            Checks if a linear inequality in the polyhedron separate any point of given points.

    """

    def __new__(cls, input_array, variables: typing.List[puan.variable] = []):
        arr = numpy.asarray(input_array).view(cls)
        arr.variables = variables
        return arr

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.variables = getattr(obj, 'variables', None)

    def _copy_attrs_to(self, target):
        target = target.view(ArraySubclass)
        try:
            target.__dict__.update(self.__dict__)
        except AttributeError:
            pass
        return target

    @property
    def A(self) -> numpy.ndarray:

        """
            Matrix 'A', as in Ax >= b.

            Returns
            -------
                out : numpy.ndarray

            Examples
            --------
                >>> ge_polyhedron = ge_polyhedron(numpy.array([
                >>>     [0,-1, 1, 0, 0],
                >>>     [0, 0,-1, 1, 0],
                >>>     [0, 0, 0,-1, 1],
                >>> ]))
                >>> ge_polyhedron.A()
                array([
                    [-1, 1, 0, 0],
                    [0,-1, 1, 0],
                    [0, 0,-1, 1],
                ])
        """
        return numpy.array(self[:, 1:])

    @property
    def b(self) -> numpy.ndarray:

        """
            Support vector 'b', as in Ax >= b.

            Returns
            -------
                out : numpy.ndarray

            Examples
            --------
                >>> ge_polyhedron = ge_polyhedron(numpy.array([
                >>>     [0,-1, 1, 0, 0],
                >>>     [0, 0,-1, 1, 0],
                >>>     [0, 0, 0,-1, 1],
                >>> ]))
                >>> ge_polyhedron.b()
                array([0,0,0])
        """
        return numpy.array(self.T[0])

    def integer_variable_indices(self) -> typing.Set[int]:

        """
            Variable indices where variable dtype is int.

            Returns
            -------
                out : Set : int

            Examples
            --------
                >>> ge_polyhedron = ge_polyhedron(numpy.array([
                >>>     [0,-1, 1, 0, 0],
                >>>     [0, 0,-1, 1, 0],
                >>>     [0, 0, 0,-1, 1],
                >>> ]), [variable("a", int), variable("b"), variable("c", int), variable("d")])
                >>> ge_polyhedron.integer_variable_indices()
                [0,2]
        """

        return set(
            map(
                operator.itemgetter(0),
                filter(
                    maz.compose(
                        functools.partial(operator.eq, int),
                        operator.attrgetter("dtype"),
                        operator.itemgetter(1)
                    ),
                    enumerate(self.variables)
                )
            )
        )

    def boolean_variable_indices(self) -> typing.Set[int]:

        """
            Variable indices where variable dtype is bool.

            Returns
            -------
                out : Set : int

            Examples
            --------
                >>> ge_polyhedron = ge_polyhedron(numpy.array([
                >>>     [0,-1, 1, 0, 0],
                >>>     [0, 0,-1, 1, 0],
                >>>     [0, 0, 0,-1, 1],
                >>> ]), [variable("a", int), variable("b"), variable("c", int), variable("d")])
                >>> ge_polyhedron.integer_variable_indices()
                [1,3]
        """

        return set(
            map(
                operator.itemgetter(0),
                filter(
                    maz.compose(
                        functools.partial(operator.eq, bool),
                        operator.attrgetter("dtype"),
                        operator.itemgetter(1)
                    ),
                    enumerate(self.variables)
                )
            )
        )

    def to_value_map(self: numpy.ndarray) -> dict:

        """
            Reduces the polyhedron into a value map.

            Returns
            -------
                out : dictionary: value -> indices

            Notes
            -----
                Since zeros are excluded in a value map, leading zero rows/columns will not be included.

            Examples
            --------
                >>> ge_polyhedron = ge_polyhedron(numpy.array([
                >>>     [0,-1, 1, 0, 0],
                >>>     [0, 0,-1, 1, 0],
                >>>     [0, 0, 0,-1, 1],
                >>> ]))
                >>> ge_polyhedron.to_value_map()
                {1: [[0, 1, 2], [2, 3, 4]], -1: [[0, 1, 2], [1, 2, 3]]}

        """
        return dict(map(lambda v:
                    (v, # Dict key
                    list(map(lambda x: x[1], enumerate(numpy.argwhere(self == v).T.tolist())))), # Dict value
                    set(self[self != 0])))

    def to_linalg(self: numpy.ndarray) -> tuple:
        """
            Assumes support vector index 0 in polyhedron
            and returns :math:`A, b` as in :math:`Ax \\ge b`

            Returns
            -------
                out : tuple
                    out[0] : A\n
                    out[1] : b

            Examples
            --------
                >>> ge = ge_polyhedron(numpy.array([
                >>>    [0,-1, 1, 0, 0],
                >>>    [0, 0,-1, 1, 0],
                >>>    [0, 0, 0,-1, 1],
                >>> ]))
                >>> ge.to_linalg()
                (array([
                    [-1, 1, 0, 0],
                    [0, -1, 1, 0],
                    [0, 0, -1, 1]]),
                array([0,0,0]))
        """
        if self.ndim < 2:
            A = self[1:].copy()
            b = self[0].copy()
        else:
            A = self[:, 1:].copy()
            b = self.T[0].copy()
        return numpy.array(A), numpy.array(b)

    def reducable_columns_approx(self: numpy.ndarray) -> numpy.ndarray:
        """
            Returns which columns are reducable under approximate condition.
            The approximate condition is that only one row of ge_polyhedron is
            considered when deducing reducable columns. By considering combination of rows
            more reducable columns might be found.

            Returns
            -------
                out : numpy.ndarray (vector)
                    Columns with positive values could be assumed.
                    Columns with negative values could be removed (not-assumed).

            See also
            --------
                reduce : Reduces matrix polyhedron by information passed in rows_vector and columns_vector.
                reduce_columns : Reducing columns from polyhedron from columns_vector where a positive number meaning *assume* and a negative number meaning *not assume*.
                reducable_rows_and_columns : Returns reducable rows and columns of given polyhedron.
                reduce_rows : Reduces rows from a rows_vector where num of rows of M equals size of rows_vector.
                reducable_rows : Returns a boolean vector indicating what rows are reducable.

            Examples
            --------
            All columns could be *assumed not* since picking any of the corresponding variable would violate the inequlity

                >>> ge = ge_polyhedron(numpy.array([[0, -1, -1, -1]]))
                >>> ge.reducable_columns_approx()
                ge_polyhedron([-2, -2, -2])

            All columns could be *assumed* since not picking any of the corresponding variable would violate the inequlity

                >>> ge = ge_polyhedron(numpy.array([[3, 1, 1, 1]]))
                >>> ge.reducable_columns_approx()
                ge_polyhedron([1, 1, 1])

            Combination of *assume* and *not assume*

                >>> ge = ge_polyhedron(numpy.array([[0, 1, 1, -3]]))
                >>> ge.reducable_columns_approx()
                ge_polyhedron([0, 0, -3])

                >>> ge = ge_polyhedron(numpy.array([[2, 1, 1, -1]]))
                >>> ge.reducable_columns_approx()
                ge_polyhedron([1, 1, -2])

            Combination of rows would give reducable column. Note that zero coulmns are kept.

                >>> ge = ge_polyhedron(numpy.array([
                >>>     [ 0,-1, 1, 0, 0, 0], # 1
                >>>     [ 0, 0,-1, 1, 0, 0], # 2
                >>>     [-1,-1, 0,-1, 0, 0], # 3 1+2+3 -> Force not variable 0
                >>> ]))
                >>> ge.reducable_columns_approx()
                ge_polyhedron([0, 0, 0, 0, 0])

            Contradicting rules

                >>> ge = ge_polyhedron(numpy.array([
                >>>     [1, 1], # Force variable 0
                >>>     [1, -1] # Force not variable 0
                >>> ]))
                >>> ge.reducable_columns_approx()
                ge_polyhedron([0])

        """
        A, b = ge_polyhedron.to_linalg(self)
        r = (A*((A*(A <= 0) + (A*(A > 0)).sum(axis=1).reshape(-1,1)) < b.reshape(-1,1))) + A*((A * (A > 0)).sum(axis=1) == b).reshape(-1,1)
        return r.sum(axis=0)

    def reduce_columns(self: numpy.ndarray, columns_vector: numpy.ndarray) -> numpy.ndarray:

        """
            Reducing columns from polyhedron from columns_vector where a positive number meaning *assume*
            and a negative number meaning *not assume*.

            Parameters
            ----------
            columns_vector : ndarray
                The polyhedron is reduced column-wise by equally many positives and negatives in columns-vector.

            Returns
            -------
                out : numpy.ndarray

            See also
            --------
                reduce : Reduces matrix polyhedron by information passed in rows_vector and columns_vector.
                reducable_rows_and_columns : Returns reducable rows and columns of given polyhedron.
                reduce_rows : Reduces rows from a rows_vector where num of rows of M equals size of rows_vector.
                reducable_rows : Returns a boolean vector indicating what rows are reducable.
                reducable_columns_approx : Returns what columns are reducable under approximate condition.


            Examples
            --------
                >>> ge_polyhedron = ge_polyhedron(numpy.array([
                >>>     [0,-1, 1, 0, 0],
                >>>     [0, 0,-1, 1, 0],
                >>>     [0, 0, 0,-1, 1],
                >>> ]))
                >>> columns_vector = numpy.array([1, 0,-1, 0]) # meaning assume index 0 and not assume index 2
                >>> ge_polyhedron.reduce_columns(columns_vector)
                ge_polyhedron([[1, 1, 0],
                               [0,-1, 0],
                               [0, 0, 1]])

        """

        A, b = ge_polyhedron.to_linalg(self)
        _b = b - (A.T*(columns_vector > 0).reshape(-1,1)).sum(axis=0)
        _A = numpy.delete(A, numpy.argwhere(columns_vector != 0).T[0], 1)
        return ge_polyhedron(numpy.append(_b.reshape(-1,1), _A, axis=1))

    def reducable_rows(self: numpy.ndarray) -> numpy.ndarray:
        """
            Returns a boolean vector indicating what rows are reducable.
            A row is reducable iff it doesn't constrain any variable.

            Returns
            -------
                out : numpy.ndarray (vector)

            See also
            --------
                reduce : Reduces matrix polyhedron by information passed in rows_vector and columns_vector.
                reduce_columns : Reducing columns from polyhedron from columns_vector where a positive number meaning *assume* and a negative number meaning *assume*.
                reducable_rows_and_columns : Returns reducable rows and columns of given polyhedron.
                reduce_rows : Reduces rows from a rows_vector where num of rows of M equals size of rows_vector.
                reducable_columns_approx : Returns what columns are reducable under approximate condition.

            Examples
            --------
            The sum of all negative numbers of the row in :math:`A` is :math:`\\ge b`, i.e.
            :math:`Ax \\ge b` will always hold, regardless of :math:`x`.

                >>> ge_polyhedron = ge_polyhedron(numpy.array([[-3, -1, -1, 1, 0]]))
                >>> ge_polyhedron.reducable_rows()
                ge_polyhedron([True])

            All elements of the row in :math:`A` is :math:`\\ge 0` and :math:`b` is :math:`\\le 0`,
            again :math:`Ax \\ge b` will always hold, regardless of :math:`x`.

                >>> ge_polyhedron = ge_polyhedron(numpy.array([[0, 1, 1, 1, 0]]))
                >>> ge_polyhedron.reducable_rows()
                ge_polyhedron([True])

        """
        A, b = ge_polyhedron.to_linalg(self)
        return (((A * (A < 0)).sum(axis=1) >= b)) + ((A >= 0).all(axis=1) & (b<=0))

    def reduce_rows(self: numpy.ndarray, rows_vector: numpy.ndarray) -> numpy.ndarray:

        """
            Reduces rows from a rows_vector where num of rows of ge_polyhedron equals
            size of rows_vector. Each row in rows_vector == 0 is kept.

            Parameters
            ----------
                rows_vector : numpy.ndarray

            Returns
            -------
                out : ge_polyhedron

            See also
            --------
                reduce : Reduces matrix polyhedron by information passed in rows_vector and columns_vector.
                reduce_columns : Reducing columns from polyhedron from columns_vector where a positive number meaning *assume* and a negative number meaning *assume*.
                reducable_rows_and_columns : Returns reducable rows and columns of given polyhedron.
                reducable_rows : Returns a boolean vector indicating what rows are reducable.
                reducable_columns_approx : Returns what columns are reducable under approximate condition.

            Examples
            --------

            >>> ge_polyhedron = ge_polyhedron(numpy.array([
            >>>     [0,-1, 1, 0, 0], # Reduce
            >>>     [0, 0,-1, 1, 0], # Keep
            >>>     [0, 0, 0,-1, 1], # Reduce
            >>> ]))
            >>> rows_vector = numpy.array([1, 0, 1])
            >>> ge_polyhedron.reduce_rows(rows_vector)
            ge_polyhedron([[0, 0, -1, 1, 0]])

            :code:`rows_vector` could be boolean

            >>> ge_polyhedron = ge_polyhedron(numpy.array([
            >>>     [0,-1, 1, 0, 0], # Reduce
            >>>     [0, 0,-1, 1, 0], # Keep
            >>>     [0, 0, 0,-1, 1], # Reduce
            >>> ]))
            >>> rows_vector = numpy.array([True, False, True])
            >>> ge_polyhedron.reduce_rows(rows_vector)
            ge_polyhedron([[0, 0, -1, 1, 0]])
        """

        return self[rows_vector == 0]

    def reducable_rows_and_columns(self: numpy.ndarray) -> tuple:

        """
            Returns reducable rows and columns of given polyhedron.

            Returns
            -------
                out : tuple
                    out[0] : a vector equal size as ge_polyhedron's row size where 1 represents a removed row and 0 represents a kept row\n
                    out[1] : a vector with equal size as ge_polyhedron's column size where a positive number represents requireds and a negative number represents forbids

            See also
            --------
                reduce : Reduces matrix polyhedron by information passed in rows_vector and columns_vector.
                reduce_rows : Reduces rows from a rows_vector where num of rows of M equals size of rows_vector.
                reducable_rows : Returns a boolean vector indicating what rows are reducable.
                reduce_columns :  Reducing columns from polyhedron from columns_vector where a positive number meaning *assume* and a negative number meaning *assume*.
                reducable_columns_approx : Returns what columns are reducable under approximate condition.

            Examples
            --------
                >>> ge_polyhedron = ge_polyhedron(np.array([
                >>>     [ 0,-1, 1, 0, 0, 0], # 1
                >>>     [ 0, 0,-1, 1, 0, 0], # 2
                >>>     [-1,-1, 0,-1, 0, 0], # 3 1+2+3 -> Force not variable 0, not identified under the approximate conditions
                >>>     [ 1, 0, 0, 0, 1, 0], # Force variable 3
                >>>     [ 0, 0, 0, 0, 0,-1], # Force not variable 4
                >>>     [ 0, 1, 1, 0, 1, 0], # Redundant rule
                >>>     [ 0, 1, 1, 0, 1,-1], # Redundant when variable 4 forced not
                >>> ]))
                >>> reducable_rows_and_columns(ge_polyhedron)
                (array([0, 0, 0, 1, 1, 1, 1]), array([ 0,  0,  0,  1, -2]))

        """

        _M = self.copy()
        red_cols = ge_polyhedron.reducable_columns_approx(_M)
        red_rows = ge_polyhedron.reducable_rows(_M) * 1
        full_cols = numpy.zeros(_M.shape[1]-1, dtype=int)
        full_rows = numpy.zeros(_M.shape[0], dtype=int)
        while red_cols.any() | red_rows.any():
            _M = ge_polyhedron.reduce_columns(_M, red_cols)
            full_cols[full_cols == 0] = red_cols

            red_rows = ge_polyhedron.reducable_rows(_M) * 1
            _M = ge_polyhedron.reduce_rows(_M, red_rows)
            full_rows[full_rows == 0] = red_rows

            red_cols = ge_polyhedron.reducable_columns_approx(_M)
            red_rows = ge_polyhedron.reducable_rows(_M) * 1

        return full_rows, full_cols

    def reduce(self: numpy.ndarray, rows_vector: numpy.ndarray=None, columns_vector: numpy.ndarray=None) -> numpy.ndarray:
        """
            Reduces matrix polyhedron by information passed in rows_vector and columns_vector.

            Parameters
            ----------
            rows_vector : numpy.ndarray (optional)
                A vector of 0's and 1's where rows matching index of value 1 are removed.

            columns_vector : numpy.ndarray (optional)
                A vector of positive and negative integers where the positive represents active selections and negative
                represents active "not" selections. The polyhedron is reduced under those assumptions.

            Returns
            -------
                out : ge_polyhedron
                    Reduced polyhedron

            See also
            --------
                reducable_rows_and_columns : Returns reducable rows and columns of given polyhedron.
                reduce_rows : Reduces rows from a rows_vector where num of rows of M equals size of rows_vector.
                reducable_rows : Returns a boolean vector indicating what rows are reducable.
                reduce_columns :  Reducing columns from polyhedron from columns_vector where a positive number meaning *assume* and a negative number meaning *assume*.
                reducable_columns_approx : Returns what columns are reducable under approximate condition.

            Examples
            --------
                >>> ge_polyhedron = ge_polyhedron(numpy.array([
                >>>     [ 0,-1, 1, 0, 0, 0, 0],
                >>>     [ 0, 0,-1, 1, 0, 0, 0],
                >>>     [-1, 0, 0,-1,-1, 0, 0],
                >>>     [ 1, 0, 0, 0, 0, 1, 1],
                >>> ]))
                >>> columns_vector = numpy.array([1,0,0,0,0,0])
                >>> ge_polyhedron.reduce(columns_vector=columns_vector)
                ge_polyhedron([[ 1, 1, 0, 0, 0, 0],
                               [ 0,-1, 1, 0, 0, 0],
                               [-1, 0,-1,-1, 0, 0],
                               [ 1, 0, 0, 0, 1, 1]
                            ])
        """
        gp = self.copy()
        if rows_vector is not None:
            gp = ge_polyhedron.reduce_rows(gp, rows_vector)
        if columns_vector is not None:
            gp = ge_polyhedron.reduce_columns(gp, columns_vector)
        return gp

    def neglectable_columns(self: numpy.ndarray, patterns: numpy.ndarray) -> numpy.ndarray:
        """
            Returns neglectable columns of given polyhedron `ge_polyhedron` based on given patterns.
            Neglectable columns are the columns which doesn't differentiate the patterns in `ge_polyhedron`
            from the patterns not in `ge_polyhedron`

            Parameters
            ----------
                patterns : numpy.ndarray
                    known patterns of variables in the polyhedron.

            Returns
            -------
                out : numpy.ndarray
                    A 1-d ndarray of length equal to the number of columns of the ge_polyhedron,
                    with ones at corresponding columns which can be neglected, zero otherwise.

            See also
            --------
                neglect_columns : Neglects columns in :math:`A` from a columns_vector.

            Notes
            -----
                This method is differentiating the patterns which are in ge_polyhedron from those that are not.
                Variables which are not in the patterns and has a positive number for any row in ge_polyhedron
                are considered non-neglectable.

            Examples
            --------
            Keep common pattern:
            ge_polyhedron with two out of three patterns. Variables 1 and 2 are not differentiating the patterns
            in ge_polyhedron from those that are not, and can therefore be neglected.

            >>> ge_polyhedron = ge_polyhedron(numpy.array([
            >>>     [-1,-1,-1, 0, 0, 0, 1],
            >>>     [-1,-1, 0,-1, 0, 0, 1],
            >>> ]))
            >>> patterns = numpy.array([
            >>>     [1, 1, 0], # Pattern in ge_polyhedron
            >>>     [0, 1, 1], # Pattern not in ge_polyhedron
            >>>     [1, 0, 1]  # Pattern in ge_polyhedron
            >>> ])
            >>> neglectable_columns(ge_polyhedron, patterns)
            ge_polyhedron([0, 1, 1, 0, 0, 0])

            Neglect common pattern:
            Variable 0 is part of all patterns and can therefore be neglected.

            >>> ge_polyhedron = ge_polyhedron(numpy.array([
            >>>         [-1,-1,-1, 0, 0, 0, 1],
            >>>         [-1,-1, 0,-1, 0, 0, 1],
            >>> ]))
            >>> patterns = numpy.array([
            >>>        [1, 1, 0],
            >>>        [1, 0, 1],
            >>>        [1, 0, 0]
            >>> ])
            >>> neglectable_columns(ge_polyhedron, patterns)
            array([1, 0, 0, 0, 0, 0])

        """
        A, b = ge_polyhedron.to_linalg(self)
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
            Neglects columns in :math:`A` from a columns_vector. The entire column for col in columns_vector :math:`>0`
            is set to :math:`0` and the support vector is updated.

            Parameters
            ----------
                columns_vector : numpy.ndarray
                    A 1-d ndarray of length equal to the number of columns of :math:`A`.
                    Sets the entire column of :math:`A` to zero for corresponding entries of columns_vector which are :math:`>0`.

            Returns
            -------
                out : ge_polyhedron
                    ge_polyhedron with neglected columns set to zero and suupport vector updated accordingly.

            See also
            --------
                neglectable_columns : Returns neglectable columns of given polyhedron `ge_polyhedron` based on given patterns.

            Examples
            --------
                >>> ge_polyhedron = ge_polyhedron(numpy.array([
                >>>     [0,-1, 1, 0, 0],
                >>>     [0, 0,-1, 1, 0],
                >>>     [0, 0, 0,-1, 1],
                >>>  ]))
                >>> columns_vector = numpy.array([1, 0, 1, 0])
                >>> neglect_columns(ge_polyhedron, columns_vector)
                ge_polyhedron([[ 1, 0, 1, 0, 0],
                               [-1, 0,-1, 0, 0],
                               [ 1, 0, 0, 0, 1]])


        """
        A, b = ge_polyhedron.to_linalg(self)
        _b = b - (A.T*(columns_vector > 0).reshape(-1,1)).sum(axis=0)
        _A = A
        _A[:, columns_vector>0] = 0
        return ge_polyhedron(numpy.append(_b.reshape(-1,1), _A, axis=1))

    def separable(self: numpy.ndarray, points: numpy.ndarray) -> numpy.ndarray:

        """
            Checks if points are separable by a hyperplane from the polyhedron.

            .. code-block::

                           / \\
                          /   \\
                point -> /  x  \\
                        /_ _ _ _\ <- polyhedron

            Parameters
            ----------
                points : numpy.ndarray
                    Points to be evaluated if separable to polyhedron by a hyperplane.

            Returns
            -------
                out : numpy.ndarray.
                    Boolean vector indicating T if separable to polyhedron by a hyperplane.

            See also
            --------
                ineq_separates_points : Checks if a linear inequality of the polyhedron separates any point of given points.

            Examples
            --------
                >>> ge_polyhedron = ge_polyhedron([[0,-2,1,1]])
                >>> ge_polyhedron.seaperable(numpy.array([
                >>>     [1, 0, 1],
                >>>     [1, 1, 1],
                >>>     [0, 0, 0]
                >>> ]))
                array([True, False, False])
        """
        if points.ndim > 2:
            return numpy.array(
                list(
                    map(self.separable, points)
                )
            )
        elif points.ndim == 2:
            A, b = ge_polyhedron.to_linalg(self)
            return numpy.array(
                (numpy.matmul(A, points.T) < b.reshape(-1,1)).any(axis=0)
            )
        elif points.ndim == 1:
            return ge_polyhedron.separable(self, numpy.array([points]))[0]

    def ineq_separate_points(self: numpy.ndarray, points: numpy.ndarray) -> numpy.ndarray:

        """
            Checks if a linear inequality in the polyhedron separates any point of given points.

            One linear equalities separates the given points here::


                           -/ \\-
                          -/   \-  x <- point
                point -> -/  x  \\-
                        -/_ _ _ _\- <- polyhedron
                          / / / /

            Parameters
            ----------
                points : numpy.ndarray

            Returns
            -------
                out : numpy.ndarray
                    boolean vector indicating T if linear inequality enfolds all points

            See also
            --------
                separable : Checks if points are inside the polyhedron.

            Notes
            -----
                This function is the inverse of ge_polyhedron.separable

            Examples
            --------
            >>> ge = ge_polyhedron(numpy.array([
            >>>     [ 0, 1, 0],
            >>>     [ 0, 1,-1],
            >>>     [-1,-1, 1]
            >>> ]))
            >>> points = numpy.array([[1, 1], [4, 2]])
            >>> ge.ineq_separate_points(points)
            array([True, True, False])

            Points in 3-d

            >>> ge = ge_polyhedron(numpy.array([
            >>>     [ 0, 1, 0,-1],
            >>>     [ 0, 1,-1, 0],
            >>>     [-1,-1, 1,-1]
            >>> ]))
            >>> points = numpy.array([
            >>>    [[1, 1, 1], [4, 2, 1]],
            >>>    [[0, 1, 0], [1, 2, 1]]
            >>> ])
            >>> ge.ineq_separate_points(points)
            array([[True, True, False],
                   [True, False, True]])

        """
        if points.ndim > 2:
            return numpy.array(
                list(
                    map(self.ineq_separate_points, points)
                )
            )
        elif points.ndim == 2:
            A, b = self.to_linalg()
            return numpy.array(
                (numpy.matmul(A, points.T) < b.reshape(-1,1)).any(axis=1)
            )
        elif points.ndim == 1:
            return ge_polyhedron.ineq_separate_points(numpy.array([points]), self)

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
            Takes an integer ndarray and truncates it such that the values of the output array totally
            shadows the inferior values of the input array. Specifically,
                - If integer ndarray is 1-D it returns a 1-D integer ndarray where each value of the output vector
                  totally shadows the values for lower input values. The lowest non-zero number, regardless of sign,
                  gets output value 1.

                - If integer ndarray is 2-D it returns a 1-D integer ndarray where each value of the output vector
                  totally shadows the values for lower input values and values of preceding rows.

                - If integer ndarray is M-D it is seen as M-2 collections of 2-D arrays which are truncated. The
                  output is of size (M-1)-D

            Returns
            -------
                out : numpy.ndarray

            Notes
            -----
            This function is used in the context of combinatorial optimization where items have different priorities among each other.
            Priorities can be positive or negative, where positive priorities means prefer and negative means disprefer.
            High number means higher priority and higher row index means higher priority than the rows below.

            Examples
            --------
                >>> integer_ndarray([1, 2, 3, 4]).truncate()
                integer_ndarray([1, 2, 4, 8])

                >>> integer_ndarray([3, 6, 2, 8]).truncate()
                integer_ndarray([2, 4, 1, 8])

                >>> integer_ndarray([-3, -6, 2, 8]).truncate()
                integer_ndarray([-2, -4, 1, 8])

                >>> integer_ndarray([1, 1, 2, 2, 2, 3]).truncate()
                integer_ndarray([ 1,  1,  3,  3,  3, 12])

                2-d

                >>> integer_ndarray([
                >>>     [ 0, 0, 0, 0, 1, 2],
                >>>     [-1,-1,-1,-1, 0, 0],
                >>>     [ 0, 0, 1, 2, 0, 0],
                >>>     [ 0, 0, 1, 0, 0, 0]
                >>> ]).truncate()
                integer_ndarray([-4, -4, 24, 12,  1,  2])

                3-d

                >>> integer_ndarray([
                >>>    [
                >>>         [ 1,  2,  2,  2,  2],
                >>>         [ 1, -2, -1,  2, -2],
                >>>         [ 1,  0,  2, -1, -1],
                >>>         [ 2,  1,  1,  0,  1]
                >>>    ], [
                >>>         [-1,  0, -1,  2,  0],
                >>>         [-1,  1, -2,  2, -1],
                >>>         [ 0, -2,  1, -2,  2],
                >>>         [ 0, -2,  2,  2,  0]
                >>>     ], [
                >>>         [ 1, -1,  0,  1,  1],
                >>>         [ 2,  2,  1,  1, -2],
                >>>         [ 2, -1, -2, -2,  0],
                >>>         [ 1, -1,  1,  0,  2]
                >>>     ]])
                integer_ndarray([
                    [ 8,  2,  2, -1,  2],
                    [-1, -4,  4,  4,  2],
                    [ 2, -2,  2,  1,  8]])
                )

        """
        if self.ndim > 2:
            return integer_ndarray(
                        list(
                            map(
                                integer_ndarray.truncate,
                                self
                            )
                        )
                    )

        elif self.ndim == 2:
            # Reduce self to only include higest priorities
            row_idxs = (self[::-1] != 0).argmax(axis=0).flatten()
            col_idxs = numpy.arange(self.shape[1])
            self_reduced = numpy.zeros(self.shape)
            self_reduced[row_idxs, col_idxs] = 1
            self_reduced = (self_reduced * self[::-1])[::-1]
            # Convert negatives to positives
            self_reduced_abs = numpy.abs(self_reduced)
            #Remove zero rows
            self_reduced_abs = self_reduced_abs[~numpy.all(self_reduced_abs == 0, axis=1)]
            if self_reduced_abs.shape[0] == 0:
                return numpy.zeros(self.shape[1], dtype=numpy.int64)
            # Prepare input to optimized bit allocation
            # Priorities of each row must be sorted and later converted back
            ufunc_inp_sorted_args = numpy.argsort(self_reduced_abs)
            self_reduced_abs_sorted = self_reduced_abs[numpy.arange(self_reduced_abs.shape[0]).reshape(-1,1), ufunc_inp_sorted_args]
            # Multiply every second row with -1, this will distinguish the priorities between the rows
            ufunc_inp = self_reduced_abs * numpy.array(list(map(lambda x: math.pow(-1, x), range(self_reduced_abs.shape[0])))).reshape(-1,1)
            # Sort priorities of each row and omit zero values
            ufunc_inp = ufunc_inp[numpy.arange(self_reduced_abs.shape[0]).reshape(-1,1), ufunc_inp_sorted_args]
            ufunc_inp = ufunc_inp[ufunc_inp != 0].flatten()
            values = npufunc.optimized_bit_allocation_64(ufunc_inp.astype(numpy.int64))
            # Update priorities with optimized bit allocation values
            self_reduced_abs_sorted[self_reduced_abs_sorted != 0] = values
            # Get reversed sorting
            ufunc_inp_sorted_args_rev = numpy.argsort(ufunc_inp_sorted_args)
            # Convert back to original sorting
            self_reduced_abs = self_reduced_abs_sorted[numpy.arange(self_reduced_abs.shape[0]).reshape(-1,1), ufunc_inp_sorted_args_rev]
            # Truncate matrix and convert back to original negatives
            truncated = self_reduced_abs.max(axis=0)
            truncated_neg = self_reduced.min(axis=0)
            truncated[truncated_neg < 0] = truncated[truncated_neg < 0] * -1
            return truncated.astype(numpy.int64)
        elif self.ndim == 1:
            return integer_ndarray.truncate(numpy.array([self]))
        else:
            return -1


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
to_value_map =                  ge_polyhedron.to_value_map
to_linalg =                     ge_polyhedron.to_linalg
reducable_columns_approx =      ge_polyhedron.reducable_columns_approx
reduce_columns =                ge_polyhedron.reduce_columns
reducable_rows =                ge_polyhedron.reducable_rows
reduce_rows =                   ge_polyhedron.reduce_rows
reducable_rows_and_columns =    ge_polyhedron.reducable_rows_and_columns
reduce =                        ge_polyhedron.reduce
neglectable_columns =           ge_polyhedron.neglectable_columns
neglect_columns =               ge_polyhedron.neglect_columns
separable =                     ge_polyhedron.separable
ineq_separate_points =          ge_polyhedron.ineq_separate_points
truncate =                      integer_ndarray.truncate
