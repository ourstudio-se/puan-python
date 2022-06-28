import numpy
import typing
import functools
import itertools
import operator
import maz
import math
import puan
import puan.npufunc as npufunc
import sys

class variable_ndarray(numpy.ndarray):
    
    def __new__(cls, input_array, variables: typing.List[puan.variable] = [], index: typing.List[typing.Union[int, puan.variable]] = []):
        arr = numpy.asarray(input_array, dtype=numpy.int64).view(cls)
        arr.variables = numpy.array(variables)
        arr.index = numpy.array(index)
        return arr

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.variables  = getattr(obj, 'variables', None)
        self.index      = getattr(obj, 'index', None)

    def _copy_attrs_to(self, target):
        target = target.view(ArraySubclass)
        try:
            target.__dict__.update(self.__dict__)
        except AttributeError:
            pass
        return target

    def integer_variable_indices(self) -> typing.Set[int]:

        """
            Variable indices where variable dtype is int.

            Returns
            -------
                out : Set : int

            Examples
            --------
                >>> ge_polyhedron = ge_polyhedron(numpy.array([
                ...     [0,-1, 1, 0, 0],
                ...     [0, 0,-1, 1, 0],
                ...     [0, 0, 0,-1, 1],
                ... ]), [puan.variable("a", 1, False), puan.variable("b", 0, False), puan.variable("c", 1, False), puan.variable("d", 0, False)])
                >>> ge_polyhedron.integer_variable_indices()
                [0, 2]
        """

        return sorted(
            map(
                operator.itemgetter(0),
                filter(
                    maz.compose(
                        functools.partial(operator.eq, 1),
                        operator.attrgetter("dtype"),
                        operator.itemgetter(1)
                    ),
                    enumerate(self.variables)
                )
            )
        )

    def construct(self, *variable_values: typing.List[typing.Tuple[str, int]], default_value: int = 0, dtype=numpy.int64) -> "variable_ndarray":

        """
            Constructs a variable_ndarray from a list of tuples of variable ID's and integers.

            Examples
            --------

            Constructing a new 1D variable ndarray shadow from this array and setting x = 5
                >>> vnd = variable_ndarray([[1,2,3], [2,3,4]], [puan.variable("x", 1, False), puan.variable("y", 1, False), puan.variable("z", 1, False)])
                >>> vnd.construct(("x", 5))
                variable_ndarray([5, 0, 0], dtype=int64)

            Constructing a new 2D variable ndarray shadow from this array and setting x0 = 5, y0 = 4 and y1 = 3
                >>> vnd = variable_ndarray([[1,2,3], [2,3,4]], [puan.variable("x", 1, False), puan.variable("y", 1, False), puan.variable("z", 1, False)])
                >>> vnd.construct([("x", 5), ("y", 4)], [("y", 3)])
                variable_ndarray([[5, 4, 0],
                                  [0, 3, 0]], dtype=int64)

            Returns
            -------
                out : variable_ndarray
        """
        if len(variable_values) == 0:
                return numpy.zeros((self.shape[1]), dtype=dtype)
        elif isinstance(variable_values[0], tuple):
            variable_indices = list(
                map(
                    self.variables.tolist().index,
                    puan.variable.from_mixed(
                        *map(
                            operator.itemgetter(0),
                            variable_values
                        )
                    )
                )
            )
            v = numpy.ones(len(self.variables), dtype=dtype) * default_value
            v[variable_indices] = list(map(operator.itemgetter(1), variable_values))
            return self.__class__(v, self.variables)
        else:
            return self.__class__(
                list(
                    itertools.starmap(
                        functools.partial(
                            self.construct,
                            default_value=default_value
                        ),
                        variable_values
                    )
                ),
                self.variables
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
                >>> ge_polyhedron(numpy.array([
                ...     [0,-1, 1, 0, 0],
                ...     [0, 0,-1, 1, 0],
                ...     [0, 0, 0,-1, 1]])).to_value_map()
                {1: [[0, 1, 2], [2, 3, 4]], -1: [[0, 1, 2], [1, 2, 3]]}

        """
        return dict(map(lambda v:
                    (v, # Dict key
                    list(map(lambda x: x[1], enumerate(numpy.argwhere(self == v).T.tolist())))), # Dict value
                    set(self[self != 0])))

class ge_polyhedron(variable_ndarray):
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

    def __new__(cls, input_array, variables: typing.List[puan.variable] = [], index: typing.List[typing.Union[int, puan.variable]] = []):
        if len(variables) == 0:
            arr = numpy.array(input_array)
            variables = list(map(functools.partial(puan.variable, dtype=0, virtual=False), range(arr.shape[arr.ndim-1])))

        if len(index) == 0:
            index = list(map(functools.partial(puan.variable, dtype=0, virtual=False), range(len(input_array))))

        return super().__new__(cls, input_array, variables=variables, index=index)

    @property
    def A(self) -> numpy.ndarray:

        """
            Matrix 'A', as in Ax >= b.

            Returns
            -------
                out : numpy.ndarray

            Examples
            --------
                >>> ge_polyhedron(numpy.array([
                ...     [0,-1, 1, 0, 0],
                ...     [0, 0,-1, 1, 0],
                ...     [0, 0, 0,-1, 1]])).A
                integer_ndarray([[-1,  1,  0,  0],
                                 [ 0, -1,  1,  0],
                                 [ 0,  0, -1,  1]])
        """
        return integer_ndarray(self[tuple([slice(None, None)]*(self.ndim-1)+[slice(1, None)])], self.variables[1:], self.index)

    @property
    def b(self) -> numpy.ndarray:

        """
            Support vector 'b', as in Ax >= b.

            Returns
            -------
                out : numpy.ndarray

            Examples
            --------
                >>> ge_polyhedron(numpy.array([
                ...     [0,-1, 1, 0, 0],
                ...     [0, 0,-1, 1, 0],
                ...     [0, 0, 0,-1, 1]])).b
                array([0, 0, 0])
        """
        return integer_ndarray(self.T[0], index=self.index)

    def boolean_variable_indices(self) -> typing.Set[int]:

        """
            Variable indices where variable dtype is bool.

            Returns
            -------
                out : Set : int

            Examples
            --------
                >>> ge_polyhedron(numpy.array([
                ...     [0,-1, 1, 0, 0],
                ...     [0, 0,-1, 1, 0],
                ...     [0, 0, 0,-1, 1]]),
                ...     [puan.variable("a", 1, False),
                ...      puan.variable("b", 0, False),
                ...      puan.variable("c", 1, False),
                ...      puan.variable("d", 0, False)]).boolean_variable_indices()
                [1, 3]
        """

        return sorted(
            map(
                operator.itemgetter(0),
                filter(
                    maz.compose(
                        functools.partial(operator.eq, 0),
                        operator.attrgetter("dtype"),
                        operator.itemgetter(1)
                    ),
                    enumerate(self.variables)
                )
            )
        )

    def construct(self, *variable_values: typing.List[str]) -> "boolean_ndarray":
        return self.A.construct(*variable_values)

    # def evaluate(self: numpy.ndarray, interpretation: "integer_ndarray") -> typing.Tuple[bool, "integer_ndarray"]:

    #     """
    #         Evaluates interpretation by updating corresponding true row values
    #         per iteration until either interpretation satisfies polyhedron OR
    #         no change was made in the iteration.

    #         Notes
    #         -----
    #         Super ge-polytope is assumed which assumes that each row index represents a column index (except row 0) 

    #         Examples
    #         --------
    #             >>> ph = ge_polyhedron([[1,1,1,1,0,0,0,0],[0,-2,0,0,1,0,1,0],[0,0,-2,0,1,1,0,0],[0,0,0,-1,0,0,1,0]],variables=puan.variable.construct(*list("0ABCdefX")),index=puan.variable.construct(*list("XABC")))
    #             >>> interpretation = ph.A.construct(list(zip(puan.variable.construct(*list("de")), [1,1])))
    #             >>> ph.evaluate(interpretation)

    #     """

    #     res = self.A.dot(interpretation) >= self.b
    #     if res.all():
    #         return True, interpretation

    #     _interpretation = self.A.construct(*zip(self.index,res*1))
    #     _interpretation += (_interpretation == 0)*interpretation
    #     _res = self.A.dot(_interpretation) >= self.b
    #     __interpretation = self.A.construct(*zip(self.index,((~res*_res)+(res*_res) >= 1)*1))
    #     __interpretation += (__interpretation == 0)*interpretation
        
    #     if (__interpretation == interpretation).all():
    #         return False, _interpretation
            
    #     return self.evaluate(__interpretation)

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
                >>> ge_polyhedron(numpy.array([
                ...     [0,-1, 1, 0, 0],
                ...     [0, 0,-1, 1, 0],
                ...     [0, 0, 0,-1, 1]])).to_linalg()
                (integer_ndarray([[-1,  1,  0,  0],
                                 [ 0, -1,  1,  0],
                                 [ 0,  0, -1,  1]]), array([0, 0, 0]))
        """
        return self.A, self.b

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

                >>> ge_polyhedron(numpy.array([[0, -1, -1, -1]])).reducable_columns_approx()
                integer_ndarray([-2, -2, -2])

            All columns could be *assumed* since not picking any of the corresponding variable would violate the inequlity

                >>> ge_polyhedron(numpy.array([[3, 1, 1, 1]])).reducable_columns_approx()
                integer_ndarray([1, 1, 1])

            Combination of *assume* and *not assume*

                >>> ge_polyhedron(numpy.array([[0, 1, 1, -3]])).reducable_columns_approx()
                integer_ndarray([ 0,  0, -3])

                >>> ge_polyhedron(numpy.array([[2, 1, 1, -1]])).reducable_columns_approx()
                integer_ndarray([ 1,  1, -2])

            Combination of rows would give reducable column. Note that zero coulmns are kept.

                >>> ge_polyhedron(numpy.array([
                ...     [ 0,-1, 1, 0, 0, 0],
                ...     [ 0, 0,-1, 1, 0, 0],
                ...     [-1,-1, 0,-1, 0, 0]])).reducable_columns_approx()
                integer_ndarray([0, 0, 0, 0, 0])

            Contradicting rules

                >>> ge_polyhedron(numpy.array([[1, 1], [1, -1]])).reducable_columns_approx()
                integer_ndarray([0])

        """
        A, b = self.to_linalg()
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
                ...     [0,-1, 1, 0, 0],
                ...     [0, 0,-1, 1, 0],
                ...     [0, 0, 0,-1, 1]]))
                >>> columns_vector = numpy.array([1, 0,-1, 0]) # meaning assume index 0 and not assume index 2
                >>> ge_polyhedron.reduce_columns(columns_vector)
                ge_polyhedron([[ 1,  1,  0],
                               [ 0, -1,  0],
                               [ 0,  0,  1]])

        """

        A, b = self.to_linalg()
        _b = b - (A.T*(columns_vector > 0).reshape(-1,1)).sum(axis=0)
        _A = numpy.delete(A, numpy.argwhere(columns_vector != 0).T[0], 1)
        return ge_polyhedron(numpy.append(_b.reshape(-1,1), _A, axis=1), self.variables[[True]+(columns_vector == 0).tolist()], self.index)

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

                >>> ge_polyhedron(numpy.array([[-3, -1, -1, 1, 0]])).reducable_rows()
                integer_ndarray([ True])

            All elements of the row in :math:`A` is :math:`\\ge 0` and :math:`b` is :math:`\\le 0`,
            again :math:`Ax \\ge b` will always hold, regardless of :math:`x`.

                >>> ge_polyhedron(numpy.array([[0, 1, 1, 1, 0]])).reducable_rows()
                integer_ndarray([ True])

        """
        A, b = ge_polyhedron(self, getattr(self, "variables", []), getattr(self, "index", [])).to_linalg()
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

            >>> rows_vector = numpy.array([1, 0, 1])
            >>> ge_polyhedron(numpy.array([
            ...     [0,-1, 1, 0, 0], # Reduce
            ...     [0, 0,-1, 1, 0], # Keep
            ...     [0, 0, 0,-1, 1]  # Reduce
            ... ])).reduce_rows(rows_vector)
            ge_polyhedron([[ 0,  0, -1,  1,  0]])

            :code:`rows_vector` could be boolean

            >>> rows_vector = numpy.array([True, False, True])
            >>> ge_polyhedron(numpy.array([
            ...     [0,-1, 1, 0, 0], # Reduce
            ...     [0, 0,-1, 1, 0], # Keep
            ...     [0, 0, 0,-1, 1]  # Reduce
            ... ])).reduce_rows(rows_vector)
            ge_polyhedron([[ 0,  0, -1,  1,  0]])
        """

        msk = numpy.array(rows_vector) == 0
        return ge_polyhedron(self[msk], getattr(self, "variables", []), self.index[msk] if hasattr(self, "index") else [])

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
                >>> ge_polyhedron(numpy.array([
                ...     [ 0,-1, 1, 0, 0, 0], # 1
                ...     [ 0, 0,-1, 1, 0, 0], # 2
                ...     [-1,-1, 0,-1, 0, 0], # 3 1+2+3 -> Force not variable 0
                ...     [ 1, 0, 0, 0, 1, 0], # Force variable 3
                ...     [ 0, 0, 0, 0, 0,-1], # Force not variable 4
                ...     [ 0, 1, 1, 0, 1, 0], # Redundant rule
                ...     [ 0, 1, 1, 0, 1,-1]  # Redundant when variable 4 forced not
                ... ])).reducable_rows_and_columns()
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
                ...     [ 0,-1, 1, 0, 0, 0, 0],
                ...     [ 0, 0,-1, 1, 0, 0, 0],
                ...     [-1, 0, 0,-1,-1, 0, 0],
                ...     [ 1, 0, 0, 0, 0, 1, 1]]))
                >>> columns_vector = numpy.array([1,0,0,0,0,0])
                >>> ge_polyhedron.reduce(columns_vector=columns_vector)
                ge_polyhedron([[ 1,  1,  0,  0,  0,  0],
                               [ 0, -1,  1,  0,  0,  0],
                               [-1,  0, -1, -1,  0,  0],
                               [ 1,  0,  0,  0,  1,  1]])
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

            >>> patterns = numpy.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
            >>> ge_polyhedron(numpy.array([
            ...     [-1,-1,-1, 0, 0, 0, 1],
            ...     [-1,-1, 0,-1, 0, 0, 1]])).neglectable_columns(patterns)
            integer_ndarray([0, 1, 1, 0, 0, 0])

            Neglect common pattern:
            Variable 0 is part of all patterns and can therefore be neglected.

            >>> patterns = numpy.array([[1, 1, 0], [1, 0, 1], [1, 0, 0]])
            >>> ge_polyhedron(numpy.array([
            ...     [-1,-1,-1, 0, 0, 0, 1],
            ...     [-1,-1, 0,-1, 0, 0, 1]])).neglectable_columns(patterns)
            integer_ndarray([1, 0, 0, 0, 0, 0])

        """
        A, b = ge_polyhedron(self).to_linalg()
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
                >>> ge_polyhedron = ge_polyhedron(numpy.array([[0,-1, 1, 0, 0], [0, 0,-1, 1, 0], [0, 0, 0,-1, 1]]))
                >>> columns_vector = numpy.array([1, 0, 1, 0])
                >>> neglect_columns(ge_polyhedron, columns_vector)
                ge_polyhedron([[ 1,  0,  1,  0,  0],
                               [-1,  0, -1,  0,  0],
                               [ 1,  0,  0,  0,  1]])
        """
        A, b = ge_polyhedron(self).to_linalg()
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
                >>> ge_polyhedron = ge_polyhedron([[0,-2, 1, 1]])
                >>> ge_polyhedron.separable(numpy.array([
                ...     [1, 0, 1],
                ...     [1, 1, 1],
                ...     [0, 0, 0]]))
                array([ True, False, False])
        """
        if points.ndim > 2:
            return numpy.array(
                list(
                    map(self.separable, points)
                )
            )
        elif points.ndim == 2:
            A, b = ge_polyhedron(self).to_linalg()
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
            >>> points = numpy.array([[1, 1], [4, 2]])
            >>> ge_polyhedron(numpy.array([
            ...     [ 0, 1, 0],
            ...     [ 0, 1,-1],
            ...     [-1,-1, 1]])).ineq_separate_points(points)
            array([False, False,  True])

            Points in 3-d

            >>> points = numpy.array([
            ...     [[1, 1, 1],
            ...      [4, 2, 1]],
            ...     [[0, 1, 0],
            ...      [1, 2, 1]]])
            >>> ge_polyhedron(numpy.array([
            ...     [ 0, 1, 0,-1],
            ...     [ 0, 1,-1, 0],
            ...     [-1,-1, 1,-1]])).ineq_separate_points(points)
            array([[False, False,  True],
                   [False,  True, False]])

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

    def construct_boolean_ndarray(self: numpy.ndarray, variables: typing.List[str]) -> "boolean_ndarray":

        """
            Constructs a boolean ndarray sharing self.A's variables (i.e. without support vector).

            Parameters
            ----------
            variables : list : str

            Returns
            -------
                out : boolean_ndarray
        """
        return boolean_ndarray.construct(self.A, variables)

    @staticmethod
    def convert_priorities(priorities: list, variables: typing.List[str]) -> "ge_polyhedron":
        """
            Creates a ge polyhedron to the problem of finding default weights given a list of priorities.
            Solving the optimization problem

                max :math:`\\sum x` \n
                s.t. :math:`Ax \\ge b`

            where A and b is given by the polyhedron, the solution :math:`x^*` gives the weights to an
            objective function which respects the priorities.

            Parameters
            ----------
                priorities : list[list[str]]
                    list of lists of size 2 where variable at index 0 has higher priority than vairable at index 1,
                    e.g. priorities = [["a", "b"]] means variable *a* has higher priority than *b*.
                variables : list[str]

            Returns
            -------
                out : ge_polyhedorn
                    Ge polyhedron with constraints to find weights for objective function respecting priorities.

            Notes
            -----
            Constraints in :math:`A` limits :math:`x` to be :math:`<0`.

            Examples
            --------
                >>> priorities = [["a", "b"]]
                >>> variables = ["a", "b"]
                >>> ge_polyhedron.convert_priorities(priorities, variables)
                ge_polyhedron([[ 1,  1, -1],
                               [ 1, -1,  0],
                               [ 1,  0, -1]])

                >>> priorities = [["a", "b"], ["c", "a"], ["e", "f"]]
                >>> variables = ["a", "b", "c", "d", "e", "f", "g"]
                >>> ge_polyhedron.convert_priorities(priorities, variables)
                ge_polyhedron([[ 1,  1, -1,  0,  0,  0,  0,  0],
                               [ 1, -1,  0,  1,  0,  0,  0,  0],
                               [ 1,  0,  0,  0,  0,  1, -1,  0],
                               [ 1, -1,  0,  0,  0,  0,  0,  0],
                               [ 1,  0, -1,  0,  0,  0,  0,  0],
                               [ 1,  0,  0, -1,  0,  0,  0,  0],
                               [ 1,  0,  0,  0, -1,  0,  0,  0],
                               [ 1,  0,  0,  0,  0, -1,  0,  0],
                               [ 1,  0,  0,  0,  0,  0, -1,  0],
                               [ 1,  0,  0,  0,  0,  0,  0, -1]])
        """
        polyhedron = numpy.zeros([len(priorities), len(variables)+1], dtype=dtype)
        polyhedron[:,0] = 1
        lez_constraint = numpy.concatenate((numpy.ones((len(variables),1)), -numpy.eye(len(variables))), axis=1)
        priority_indices = list(map(lambda x: [variables.index(x[0]) + 1, variables.index(x[1]) + 1], priorities))
        def _priorities_constraints(x):
            x[0][x[1][0]] = 1
            x[0][x[1][1]] = -1
            return x
        list(map(_priorities_constraints, zip(polyhedron,priority_indices)))
        variables = list(map(lambda x: (x, int), variables))
        return ge_polyhedron(numpy.concatenate((polyhedron, lez_constraint), axis=0), variables)

class integer_ndarray(variable_ndarray):
    """
        A numpy.ndarray sub class with only integers in it.

        Attributes
        ----------
        See numpy.array

        Methods
        -------
        compress
            Takes an integer ndarray and compresses it into a vector.
        to_value_map
            reduces the matrix into a value map.
        from_list
            Turns a list (or list of list) of strings into an integer vector. (static)
    """


    def reduce2d(self: numpy.ndarray, method: typing.Literal["first", "last"]="first", axis: int=0) -> numpy.ndarray:
        """
            Reduces integer ndarray to only keeping one value of the given axis (default is 0) according to the provided method.
        """
        if not self.ndim == 2:
            raise ValueError(f"ndarray must be 2-D, is {self.ndim}-D")
        self = numpy.swapaxes(self, 0, axis)
        col_idxs = numpy.arange(self.shape[1])
        self_reduced = numpy.zeros(self.shape)
        if method == "first":
            row_idxs = (self != 0).argmax(axis=0).flatten()
            self_reduced[row_idxs, col_idxs] = 1
            self_reduced = (self_reduced * self)
        elif method == "last":
            row_idxs = (self[::-1] != 0).argmax(axis=0).flatten()
            self_reduced[row_idxs, col_idxs] = 1
            self_reduced = (self_reduced * self[::-1])[::-1]
        else:
            raise ValueError(f"Method not recognized, must be either 'first' or 'last', is {method}.")
        return numpy.swapaxes(self_reduced, 0, axis)

    def ranking(self: numpy.ndarray) -> numpy.ndarray:
        if self.ndim > 1:
            return list(map(integer_ndarray.ranking, self))
        elif self.ndim == 1:
            idx_sorted = numpy.argsort(self)
            self = self[idx_sorted]
            current_ranking = 1 if (self[0] > 0) else 0
            current_val = self[0]
            for i in range(len(self)):
                if not current_val == self[i]:
                    current_ranking += 1
                    current_val = self[i]
                self[i] = current_ranking
            rev = numpy.argsort(idx_sorted)
            return self[rev]
        else:
            raise ValueError("Dimension out of bounds")

    def ndint_compress(self: numpy.ndarray, method: typing.Literal["first", "last", "min", "max", "prio", "shadow"]="min", axis: int=None, dtype=numpy.int64) -> numpy.ndarray:

        """
            Takes an integer ndarray and compresses it into a vector under different conditions given by `method`.

            Parameters
            ----------
                method : {'first', 'last', 'min', 'max', 'prio', 'shadow'}, optional
                    The method used to compress the `integer ndarray`. The following methods are available (default is 'min')

                    - 'min' Takes the minimum non-zero value
                    - 'max' Takes the maximum value
                    - 'last' Takes the last value
                    - 'prio' Treats the values as prioritizations with prioritizations increasing along the axis.
                    - 'shadow' Treats the values as prioritizations as for 'prio' but the result value of a higher prioritization totally shadows lower priorities
                axis : {None, int}, optional
                    Axis along which to perform the compressing. If None, the data array is first flattened.

            Returns
            -------
                out : numpy.ndarray
                    If input integer ndarray is input array is M-D the returned array is (M-1)-D

            Examples
            --------
            Method 'min'
                >>> integer_ndarray([[1, 2], [0, 3]]).ndint_compress()
                integer_ndarray([1, 2, 0, 3])

                >>> integer_ndarray([
                ...     [1, 2],
                ...     [0, 3]]).ndint_compress(method='min', axis=0)
                integer_ndarray([1, 2])

                >>> integer_ndarray([
                ...     [1, 2],
                ...     [0, 3]]).ndint_compress(method='min', axis=1)
                integer_ndarray([1, 3])

            Method 'max'
                >>> integer_ndarray([
                ...     [1, 2],
                ...     [0, 3]]).ndint_compress(method='max', axis=0)
                integer_ndarray([1, 3])

                >>> integer_ndarray([
                ...     [1, 2],
                ...     [0, 3]]).ndint_compress(method='max', axis=1)
                integer_ndarray([2, 3])

            Method 'first'

                >>> integer_ndarray([
                ...     [1, 2, 0, 0],
                ...     [0, 3, 4, 0],
                ...     [5, 0, 0, 6]]).ndint_compress(method='first', axis=0)
                integer_ndarray([1, 2, 4, 6])

                >>> integer_ndarray([
                ...     [1, 2, 0, 0],
                ...     [0, 3, 4, 0],
                ...     [5, 0, 0, 6]]).ndint_compress(method='first', axis=1)
                integer_ndarray([1, 3, 5])

            Method 'last'

                >>> integer_ndarray([
                ...     [1, 2, 0, 0],
                ...     [0, 3, 4, 0],
                ...     [5, 0, 0, 6]]).ndint_compress(method='last', axis=0)
                integer_ndarray([5, 3, 4, 6])

                >>> integer_ndarray([
                ...     [1, 2, 0, 0],
                ...     [0, 3, 4, 0],
                ...     [5, 0, 0, 6]]).ndint_compress(method='last', axis=1)
                integer_ndarray([2, 4, 6])

            Method 'prio'
                >>> integer_ndarray([1, 2, 1, 0, 4, 4, 6]).ndint_compress(method='prio')
                integer_ndarray([1, 2, 1, 0, 3, 3, 4])

                >>> integer_ndarray([
                ...     [1, 2, 0, 0],
                ...     [0, 3, 4, 0],
                ...     [5, 0, 0, 6]]).ndint_compress(method='prio', axis=0)
                integer_ndarray([3, 1, 2, 4])

                >>> integer_ndarray([
                ...     [1, 2, 0, 0],
                ...     [0, 3, 4, 0],
                ...     [5, 0, 0, 6]]).ndint_compress(method='prio', axis=1)
                integer_ndarray([1, 2, 3])

            Method 'shadow'
                >>> integer_ndarray([1, 2, 1, 0, 4, 4, 6]).ndint_compress(method='shadow')
                integer_ndarray([ 1,  3,  1,  0,  6,  6, 18])

                >>> integer_ndarray([
                ...     [1, 2, 0, 0],
                ...     [0, 3, 4, 0],
                ...     [5, 0, 0, 6]]).ndint_compress(method='shadow', axis=0)
                integer_ndarray([4, 1, 2, 8])

                >>> integer_ndarray([
                ...     [1, 2, 0, 0],
                ...     [0, 3, 4, 0],
                ...     [5, 0, 0, 6]]).ndint_compress(method='shadow', axis=1)
                integer_ndarray([1, 2, 4])

        """
        if not isinstance(axis, int):
            self = integer_ndarray([self.flatten()])
            axis=0
        if method == "last":
            func = lambda x: x[numpy.max(numpy.argwhere(x!=0), axis=0)]
            return numpy.add.reduce(numpy.apply_along_axis(func, axis=axis, arr=self), axis=axis)
        elif method == "first":
            self = numpy.swapaxes(self, 0, axis)
            if self.ndim > 2:
                return numpy.swapaxes(integer_ndarray(
                            list(
                                map(
                                    lambda x: integer_ndarray.ndint_compress(x, method=method, axis=0, dtype=dtype),
                                    self
                                )
                            )
                        ), 0, axis)
            elif self.ndim == 2:
                self = self[numpy.argmax(self!=0, axis=0),numpy.arange(self.shape[1])]
                return numpy.swapaxes(self, 0, axis-1)
            elif self.ndim == 1:
                return numpy.swapaxes(self, 0, axis-1)
            else:
                raise ValueError("Dimension out of bound")
        elif method == "min":
            tmp = self.copy()
            tmp[tmp==0] = sys.maxsize
            tmp = numpy.min(tmp, axis=axis)
            tmp[(self == 0).all(axis=axis)] = 0
            return tmp
        elif method == "max":
            return numpy.max(self, axis=axis)
        elif method == "prio":
            self = numpy.swapaxes(self, 0, axis)
            if self.ndim > 2:
                return numpy.swapaxes(integer_ndarray(
                            list(
                                map(
                                    lambda x: integer_ndarray.ndint_compress(x, method=method, axis=0, dtype=dtype),
                                    self
                                )
                            )
                        ), 0, axis)
            elif self.ndim == 2:
                self_reduced = integer_ndarray(self).reduce2d(method="last", axis=0)
                self_reduced = integer_ndarray(self_reduced.ranking())
                self_reduced = self_reduced + ((self_reduced.T>0) * numpy.concatenate(([0], (numpy.cumsum(self_reduced.max(axis=1)))))[:-1]).T
                return self_reduced.ndint_compress(method="first", axis=0, dtype=dtype)
            elif self.ndim == 1:
                return self.ranking()
            else:
                raise ValueError("Dimension out of bounds")


        elif method == "shadow":
            self = numpy.swapaxes(self, 0, axis)
            if self.ndim > 2:
                return numpy.swapaxes(integer_ndarray(
                            list(
                                map(
                                    lambda x: integer_ndarray.ndint_compress(x, method=method, axis=0, dtype=dtype),
                                    self
                                )
                            )
                        ), 0, axis)
            elif self.ndim == 2:
                # Reduce self to only include last values of columns
                self_reduced = integer_ndarray(self).reduce2d(method="last", axis=0).astype(self.dtype)
                # Convert negatives to positives
                self_reduced_abs = numpy.abs(self_reduced)
                #Remove zero rows
                self_reduced_abs = self_reduced_abs[~numpy.all(self_reduced_abs == 0, axis=1)]
                if self_reduced_abs.shape[0] == 0:
                    return numpy.zeros(self.shape[1], dtype=self.dtype)
                # Prepare input to optimized bit allocation
                # Values of each row must be sorted and later converted back
                ufunc_inp_sorted_args = numpy.argsort(self_reduced_abs)
                self_reduced_abs_sorted = self_reduced_abs[numpy.arange(self_reduced_abs.shape[0]).reshape(-1,1), ufunc_inp_sorted_args]
                # Multiply every second row with -1, this will distinguish the rows
                ufunc_inp = self_reduced_abs * numpy.array(list(map(lambda x: math.pow(-1, x), range(self_reduced_abs.shape[0]))), dtype=self.dtype).reshape(-1,1)
                # Sort values of each row and omit zero values
                ufunc_inp = ufunc_inp[numpy.arange(self_reduced_abs.shape[0]).reshape(-1,1), ufunc_inp_sorted_args]
                ufunc_inp = ufunc_inp[ufunc_inp != 0].flatten()
                values = npufunc.optimized_bit_allocation_64(ufunc_inp.astype(numpy.int64))
                # Update values with optimized bit allocation values
                self_reduced_abs_sorted[self_reduced_abs_sorted != 0] = values
                # Get reversed sorting
                ufunc_inp_sorted_args_rev = numpy.argsort(ufunc_inp_sorted_args)
                # Convert back to original sorting
                self_reduced_abs = self_reduced_abs_sorted[numpy.arange(self_reduced_abs.shape[0]).reshape(-1,1), ufunc_inp_sorted_args_rev]
                # Truncate matrix and convert back to original negatives
                compressed = self_reduced_abs.max(axis=0)
                compressed_neg = self_reduced.min(axis=0)
                compressed[compressed_neg < 0] = compressed[compressed_neg < 0] * -1
                return numpy.swapaxes(compressed, 0, axis-1)
            elif self.ndim == 1:
                return integer_ndarray.ndint_compress(numpy.array([self], dtype=dtype), method=method, axis=0, dtype=dtype)
            else:
                raise ValueError("Dimension out of bounds, is {self.ndim}")
        else:
            print("Method not recognized")
            return

    def get_neighbourhood(self: numpy.ndarray, method: typing.Literal["all", "addition", "subtraction"]="all", delta: typing.Union[int, numpy.ndarray]=1) -> "integer_ndarray":
        """
            Computes all neighbourhoods to an integer ndarray. A neighbourhood to the integer ndarray :math:`x` is defined as the
            integer ndarray :math:`y` which for one variable differs by *delta*, the values for the other variables are identical.

            Parameters
            ----------
                method : {'all', 'addition', 'subtraction'}, optional
                    The method used to compute the neighbourhoods to the integer ndarray. The following methods are available (default is 'all')

                    - 'all' computes all neighbourhoods, *addition* and *subtraction*.
                    - 'addition' computes all neighbourhoods with *delta* added
                    - 'subtraction' computes all neighbourhoods with *delta* subtracted

                delta : int or numpy.ndarray, optional
                    The value added or subtracted for each variable to reach the neighbourhood, default is 1.
                    If delta is given as an array it is interpreted as different deltas for each variable.

            Returns
            -------
                out : integer_ndarray
                    If input dimension is M-D the returned integer ndarray is M+1-D

            Examples
            --------
                >>> x = integer_ndarray([0,0,0])
                >>> x.get_neighbourhood()
                array([[ 1,  0,  0],
                       [ 0,  1,  0],
                       [ 0,  0,  1],
                       [-1,  0,  0],
                       [ 0, -1,  0],
                       [ 0,  0, -1]])

                >>> x = integer_ndarray([[0,0,0], [0,0,0]])
                >>> x.get_neighbourhood()
                array([[[ 1,  0,  0],
                        [ 0,  1,  0],
                        [ 0,  0,  1],
                        [-1,  0,  0],
                        [ 0, -1,  0],
                        [ 0,  0, -1]],
                <BLANKLINE>
                       [[ 1,  0,  0],
                        [ 0,  1,  0],
                        [ 0,  0,  1],
                        [-1,  0,  0],
                        [ 0, -1,  0],
                        [ 0,  0, -1]]])

                >>> x = integer_ndarray([0, 1, 2])
                >>> x.get_neighbourhood(delta=3)
                array([[ 3,  1,  2],
                       [ 0,  4,  2],
                       [ 0,  1,  5],
                       [-3,  1,  2],
                       [ 0, -2,  2],
                       [ 0,  1, -1]])
                >>> x.get_neighbourhood(delta=numpy.array([1, 2, 3]))
                array([[ 1,  1,  2],
                       [ 0,  3,  2],
                       [ 0,  1,  5],
                       [-1,  1,  2],
                       [ 0, -1,  2],
                       [ 0,  1, -1]])

                >>> x = integer_ndarray([[0,0,0], [0,0,0]])
                >>> x.get_neighbourhood(method="addition")
                integer_ndarray([[[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]],
                <BLANKLINE>
                                 [[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]]])

                >>> x = integer_ndarray([[0,0,0], [0,0,0]])
                >>> x.get_neighbourhood(method="subtraction")
                integer_ndarray([[[-1,  0,  0],
                                  [ 0, -1,  0],
                                  [ 0,  0, -1]],
                <BLANKLINE>
                                 [[-1,  0,  0],
                                  [ 0, -1,  0],
                                  [ 0,  0, -1]]])

        """
        nvars = len(self.variables)
        if not method == "subtraction":
            inc_neighbourhood = numpy.tile(self, nvars).reshape(list(self.shape)+[nvars]) +  numpy.tile(delta * numpy.eye(nvars, dtype=numpy.int64), (list(self.shape)[:-1] + [1])).reshape(list(self.shape)+[nvars])
            if method == "addition":
                return inc_neighbourhood
        if not method == "addition":
            dec_neighbourhood = numpy.tile(self, nvars).reshape(list(self.shape)+[nvars]) -  numpy.tile(delta * numpy.eye(nvars, dtype=numpy.int64), (list(self.shape)[:-1] + [1])).reshape(list(self.shape)+[nvars])
            if method == "subtraction":
                return dec_neighbourhood
        return numpy.concatenate((inc_neighbourhood, dec_neighbourhood), axis=self.ndim-1)


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
                >>> variables = ["a","c","b"]
                >>> context = ["a","b","c","d"]
                >>> integer_ndarray.from_list(variables, context)
                integer_ndarray([1, 3, 2, 0])

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

class boolean_ndarray(variable_ndarray):
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
                boolean_ndarray([1, 1, 1, 0])

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

    def construct(self, *variable_values: typing.List[str]) -> "boolean_ndarray":

        if isinstance(variable_values[0], str):
            return boolean_ndarray(variable_ndarray.construct(self, list(map(lambda x: (x, 1), variable_values))), self.variables)
        else:
            return boolean_ndarray(
                list(
                    map(
                        self.construct,
                        variable_values
                    )
                ),
                self.variables
            )

    def to_list(self, skip_virtual_variables: bool = False) -> typing.List[puan.variable]:

        """
            Returns a list of variables for each value in self equal 1.

            Returns
            -------
                out : list : puan.variable
        """

        if self.ndim == 1:
            return list(
                filter(
                    lambda variable: not (skip_virtual_variables and variable.virtual),
                    numpy.array(self.variables)[self == 1].tolist()
                )
            )
        else:
            return list(
                map(
                    functools.partial(
                        boolean_ndarray.to_list,
                        skip_virtual_variables=skip_virtual_variables,
                    ),
                    self
                )
            )

    def get_neighbourhood(self: numpy.ndarray, method: typing.Literal["on_off", "on", "off"]="on_off") -> "boolean_ndarray":
        """
            Computes all neighbourhoods to a boolean ndarray. A neighbourhood to the boolean ndarray :math:`x` is defined as the
            boolean ndarray :math:`y` which for one and only one variable in :math:`x` is the complement.

            Parameters
            ----------
                method : {'on_off', 'on', 'off'}, optional
                    The method used to compute the neighbourhoods to the boolean ndarray. The following methods are available (default is 'on_off')

                    - 'on_off' the most natural way to define neighbourhoods, if a variable is True in the input it is False in its neighbourhood and vice versa.
                    - 'on' do not include neighbourhoods with 'off switches', i.e. if a variable is True there is no neighbourhood to this variable.
                    - 'off' do not include neighbourhoods with 'on switches', i.e. if a variable is False there is no neighbourhood to this variable.

            Returns
            -------
                out : integer_ndarray
                    If input dimension is M-D the returned integer ndarray is M+1-D

            Examples
            --------
                >>> x = boolean_ndarray([1, 0, 0, 1])
                >>> x.get_neighbourhood(method="on_off")
                boolean_ndarray([[False, False, False,  True],
                                 [ True,  True, False,  True],
                                 [ True, False,  True,  True],
                                 [ True, False, False, False]])

                >>> x = boolean_ndarray([1, 0, 0, 1])
                >>> x.get_neighbourhood(method="on")
                boolean_ndarray([[ True,  True, False,  True],
                                 [ True, False,  True,  True]])

                >>> x = boolean_ndarray([1, 1, 1, 1])
                >>> x.get_neighbourhood(method="on")
                boolean_ndarray([], shape=(0, 4), dtype=bool)

                >>> x = boolean_ndarray([1, 0, 0, 1])
                >>> x.get_neighbourhood(method="off")
                boolean_ndarray([[False, False, False,  True],
                                 [ True, False, False, False]])

                >>> x = boolean_ndarray([0, 0, 0, 0])
                >>> x.get_neighbourhood(method="off")
                boolean_ndarray([], shape=(0, 4), dtype=bool)
        """
        nvars = len(self.variables)
        if method == "on_off":
            _res = numpy.logical_xor(numpy.tile(self, nvars).reshape(list(self.shape)+[nvars]), numpy.tile(numpy.eye(nvars, dtype=numpy.int64), (list(self.shape)[:-1] + [1])).reshape(list(self.shape)+[nvars]))
        elif method == "on":
            _res = numpy.logical_or(numpy.tile(self, nvars).reshape(list(self.shape)+[nvars]), numpy.tile(numpy.eye(nvars, dtype=numpy.int64), (list(self.shape)[:-1] + [1])).reshape(list(self.shape)+[nvars]))
        elif method == "off":
            _res = numpy.logical_and(numpy.tile(self, nvars).reshape(list(self.shape)+[nvars]), numpy.tile(numpy.ones(nvars, dtype=numpy.int64) - numpy.eye(nvars, dtype=numpy.int64), (list(self.shape)[:-1] + [1])).reshape(list(self.shape)+[nvars]))
        else:
            print("Method not recognized")
            return
        return _res[~(_res==self).all(axis=self.ndim-1)]
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
ndint_compress =                integer_ndarray.ndint_compress
