import base64
import gzip
import pickle
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
import puan_rspy as prs

from collections import Counter

class variable_ndarray(numpy.ndarray):
    """
        A numpy.ndarray sub class which ties variables to the indices of the ndarray.

        Attributes
        ----------
        See numpy.array

        Methods
        -------
        variable_indices
            Returns the indices of variable of the input type.
        boolean_variable_indices
            Returns the indices of boolean variables
        integer_variable_indices
            Returns the indices of integer variables
        construct
            Constructs a variable_ndarray from a list of tuples of variable IDs and integers.
        to_value_map
            Reduces the polyhedron into a value map.
        

    """
    def __new__(cls, input_array, variables: typing.List[puan.variable] = [], index: typing.List[typing.Union[int, puan.variable]] = [], dtype=numpy.int64):
        arr = numpy.asarray(input_array, dtype=dtype).view(cls)
        if len(variables) == 0:
            variables = variable_ndarray._default_variable_list(arr.shape[arr.ndim-1])

        if len(index) == 0:
            index = list(map(functools.partial(puan.variable, bounds=(0,1)), range(arr.shape[arr.ndim-2])))

        arr.variables = numpy.array(variables)
        arr.index = numpy.array(index)
        if (arr.index.size, arr.variables.size) != (arr.shape[arr.ndim-2], arr.shape[arr.ndim-1]):
            raise ValueError(f"array shape mismatch: array shape is {arr.shape} while length of index and variables are {(arr.index.size, arr.variables.size)}")

        return arr

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.variables  = getattr(obj, 'variables', None)
        self.index      = getattr(obj, 'index', None)

    def _copy_attrs_to(self, target):
        target = target.view(self.__class__)
        try:
            target.__dict__.update(self.__dict__)
        except AttributeError:
            pass
        return target

    @staticmethod
    def _default_variable_list(n, default_bounds_type: str = "bool") -> list:

        """
            Generates a default list of variables with first variable
            as the support vector variable default from `puan.variable`.

            Returns
            -------
                out : list
        """
        return list(
            itertools.chain(
                map(
                    lambda _: puan.variable.support_vector_variable(),
                    range(
                        numpy.clip(
                            n,
                            a_min=0,
                            a_max=1,
                        )
                    )
                ),
                map(
                    functools.partial(
                        puan.variable, 
                        dtype=default_bounds_type,
                    ), 
                    range(1, n)
                )
            )
        )

    def variable_indices(self, variable_type: int) -> numpy.ndarray:

        """
            Variable indices of variable type 0 (bool) or 1 (int).

            Returns
            -------
                out : numpy.ndarray 
        """
        is_bool = 1*(variable_type==0)
        return numpy.array(
            sorted(
                map(
                    operator.itemgetter(0),
                    filter(
                        lambda x: 1*(x[1].bounds.as_tuple() != (0,1)) + is_bool == 1,
                        enumerate(self.variables)
                    )
                )
            )
        )

    @property
    def boolean_variable_indices(self) -> numpy.ndarray:

        """
            Variable indices where variable dtype is bool.

            Returns
            -------
                out : numpy.ndarray

            Examples
            --------
                >>> ge_polyhedron = ge_polyhedron(numpy.array([
                ...     [0,-1, 1, 0, 0],
                ...     [0, 0,-1, 1, 0],
                ...     [0, 0, 0,-1, 1],
                ... ]), [puan.variable("0",(-10,10)), puan.variable("a",(0,10)), puan.variable("b"), puan.variable("c",(-2,2)), puan.variable("d")])
                >>> ge_polyhedron.boolean_variable_indices
                array([2, 4])
        """

        return self.variable_indices(0)

    @property
    def integer_variable_indices(self) -> numpy.ndarray:

        """
            Variable indices where variable dtype is int.

            Returns
            -------
                out : numpy.ndarray

            Examples
            --------
                >>> ge_polyhedron = ge_polyhedron(numpy.array([
                ...     [0,-1, 1, 0, 0],
                ...     [0, 0,-1, 1, 0],
                ...     [0, 0, 0,-1, 1],
                ... ]), [puan.variable("0",(-10,10)), puan.variable("a",(0,10)), puan.variable("b"), puan.variable("c",(-2,2)), puan.variable("d")])
                >>> ge_polyhedron.integer_variable_indices
                array([0, 1, 3])
        """

        return self.variable_indices(1)

    def construct(self, *variable_values: typing.List[typing.Tuple[str, int]], default_value: int = 0, dtype=numpy.int64) -> numpy.ndarray:

        """
            Constructs a variable_ndarray from a list of tuples of variable IDs and integers.

            Examples
            --------

            Constructing a new 1D variable ndarray shadow from this array and setting x = 5
                >>> vnd = variable_ndarray([[1,2,3], [2,3,4]], [puan.variable("x"), puan.variable("y"), puan.variable("z")])
                >>> vnd.construct(("x", 5))
                array([5, 0, 0])

            Constructing a new 2D variable ndarray shadow from this array and setting x0 = 5, y0 = 4 and y1 = 3
                >>> vnd = variable_ndarray([[1,2,3], [2,3,4]], [puan.variable("x"), puan.variable("y"), puan.variable("z")])
                >>> vnd.construct([("x", 5), ("y", 4)], [("y", 3)])
                array([[5, 4, 0],
                       [0, 3, 0]])

            Notes
            -----
            If a variable in `variable_values` is not in self.variables, then it will be ignored.

            Returns
            -------
                out : variable_ndarray
        """
        variable_ids = list(map(lambda x: x.id, self.variables))
        if len(variable_values) == 0:
            return numpy.ones((self.shape[1]), dtype=dtype)*default_value
        elif isinstance(variable_values[0], tuple):
            filtered_variable_values = list(
                filter(
                    lambda x: x[0] in variable_ids,
                    variable_values
                )
            )
            variable_indices = list(
                map(
                    variable_ids.index,
                    puan.variable.from_mixed(
                        *map(
                            operator.itemgetter(0),
                            filtered_variable_values
                        )
                    )
                )
            )
            v = numpy.ones(len(self.variables), dtype=dtype) * default_value
            v[variable_indices] = list(map(operator.itemgetter(1), filtered_variable_values))
            return v
        else:
            return numpy.array(
                list(
                    itertools.starmap(
                        functools.partial(
                            self.construct,
                            default_value=default_value
                        ),
                        variable_values
                    )
                ),
                dtype=dtype
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
        row_bounds
            Returns the row equation bounds including bias.
        row_distribution
            Returns a distribution of all combinations for each row and their values.
        row_stretch
            Shows the proportion of the number of value spots for each row equation with respect to the number of combinations from active variable bounds.
        row_stretch_int
            Row bounds range from one number to another.
        neglectable_columns
            Returns neglectable columns of given polyhedron `ge_polyhedron`.
        neglect_columns
            Neglects columns in :math:`A` from a columns_vector.
        separable
            Checks if points are inside the polyhedron.
        ineq_separate_points
            Checks if a linear inequality in the polyhedron separate any point of given points.

    """

    def __new__(cls, input_array, variables: typing.List[puan.variable] = [], index: typing.List[typing.Union[int, puan.variable]] = [], dtype=numpy.int64):
        return super().__new__(cls, input_array, variables=variables, index=index, dtype=dtype)

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
    def A_max(self) -> numpy.array:

        """
            Returns the maximum coefficient value based on variable's initial bounds.

            Returns
            -------
                out : numpy.ndarray
        """
        init = self.column_bounds()
        return (init[0]*(self.A<0)*self.A)+(init[1]*(self.A>0)*self.A)

    @property
    def A_min(self) -> numpy.array:

        """
            Returns the minimum coefficient value based on variable's initial bounds.

            Raises
            ------
                Exception
                    If matrix A is empty.

            Returns
            -------
                out : numpy.ndarray
        """
        init = self.column_bounds()
        if init.size == 0:
            raise Exception(f"Matrix A in polyhedron is empty")

        return (init[0]*(self.A>0)*self.A)+(init[1]*(self.A<0)*self.A)

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
        return numpy.array(self.T[0])

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

            Rows with values :math:`\\neq0` for any integer variable are neglected.

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
                array([0., 0., 0.])

            All columns could be *assumed* since not picking any of the corresponding variable would violate the inequlity

                >>> ge_polyhedron(numpy.array([[3, 1, 1, 1]])).reducable_columns_approx()
                array([1., 1., 1.])

            Combination of *assume* and *not assume*

                >>> ge_polyhedron(numpy.array([[2, 1, 1, -1]])).reducable_columns_approx()
                array([1., 1., 0.])

        """
        A, b = ge_polyhedron(self, getattr(self, "variables", []), getattr(self, "index", [])).to_linalg()
        res = numpy.nan * numpy.zeros(A.shape[1], dtype=float)
        lb, ub = self.tighten_column_bounds()
        eq_bounds = lb == ub
        if eq_bounds.size > 0:
            res[eq_bounds] = lb[eq_bounds]
        return res

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
                >>> columns_vector = numpy.array([1, numpy.nan, 0, numpy.nan]) # meaning assume index 0 and not assume index 2
                >>> ge_polyhedron.reduce_columns(columns_vector)
                ge_polyhedron([[ 1,  1,  0],
                               [ 0, -1,  0],
                               [ 0,  0,  1]])

        """

        A, b = self.to_linalg()
        active_columns = ~numpy.isnan(columns_vector)
        _b = b-(A[:, active_columns]*columns_vector[active_columns]).sum(axis=1)
        _A = numpy.delete(A, numpy.argwhere(active_columns).T[0], 1)
        res = ge_polyhedron(numpy.append(_b.reshape(-1,1), _A, axis=1), self.variables[[True]+numpy.isnan(columns_vector).tolist()], self.index).astype(numpy.int64)
        return res

    def reducable_rows(self: numpy.ndarray) -> numpy.ndarray:
        """
            Returns a boolean vector indicating what rows are reducable.
            A row is reducable iff it doesn't constrain any variable.

            Rows with values :math:`\\neq0` for any integer variable are neglected.

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
        return self.A_min.sum(axis=1) >= self.b

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
                    out[1] : a vector with equal size as ge_polyhedron's column size where nan numbers represent a no-choice.

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
                ...     [ 0,-1, 1, 0, 0, 0],
                ...     [ 0, 0,-1, 1, 0, 0],
                ...     [-1,-1, 0,-1, 0, 0],
                ...     [ 1, 0, 0, 0, 1, 0],
                ...     [ 0, 0, 0, 0, 0,-1],
                ...     [ 0, 1, 1, 0, 1, 0],
                ...     [ 0, 1, 1, 0, 1,-1] 
                ... ])).reducable_rows_and_columns()
                (array([0, 0, 0, 1, 1, 1, 1]), array([nan, nan, nan,  1.,  0.]))

        """

        _M = self.copy()
        red_cols = ge_polyhedron.reducable_columns_approx(_M)
        red_rows = ge_polyhedron.reducable_rows(_M) * 1
        full_cols = numpy.zeros(_M.A.shape[1], dtype=int)*numpy.nan
        full_rows = numpy.zeros(_M.shape[0], dtype=int)
        while (~numpy.isnan(red_cols)).any() | red_rows.any():
            _M = ge_polyhedron.reduce_columns(_M, red_cols)
            full_cols[numpy.isnan(full_cols)] = red_cols

            if _M.shape[1] <= 1:
                break

            red_rows = ge_polyhedron.reducable_rows(_M) * 1
            _M = ge_polyhedron.reduce_rows(_M, red_rows)
            full_rows[full_rows == 0] = red_rows

            if _M.shape[0] == 0:
                break

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
                A vector of positive, negative and NaN floats. The NaN's represent a no-choice.

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
                >>> columns_vector = numpy.array([1,numpy.nan,numpy.nan,numpy.nan,numpy.nan,numpy.nan])
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
    
    def ineqs_satisfied(self: numpy.ndarray, points: numpy.ndarray) -> numpy.ndarray:
        """
            Checks if the linear inequalities in the polyhedron are satisfied given the points, i.e. :math:`A \\cdot p \\ge b`, given :math:`p` in points.

            Parameters
            ----------
                points : numpy.ndarray

            Returns
            -------
                out : numpy.ndarray
                    boolean vector indicating T if linear inequality is satisfied given the point

            Examples
            --------
            >>> points = numpy.array([[1, 1], [4, 2]])
            >>> ge_polyhedron(numpy.array([
            ...     [ 0, 1, 0],
            ...     [ 0, 1,-1],
            ...     [-1,-1, 1]])).ineqs_satisfied(points)
            boolean_ndarray([1, 0])

            Points in 3-d

            >>> points = numpy.array([
            ...     [[1, 1, 1],
            ...      [4, 2, 1]],
            ...     [[0, 1, 0],
            ...      [1, 2, 1]]])
            >>> ge_polyhedron(numpy.array([
            ...     [ 0, 1, 0,-1],
            ...     [ 0, 1,-1, 0],
            ...     [-1,-1, 1,-1]])).ineqs_satisfied(points)
            array([[1, 0],
                   [0, 0]])

        """
        if points.ndim > 2:
            return numpy.array(
                list(
                    map(self.ineqs_satisfied, points)
                )
            )
        elif points.ndim == 2:
            return boolean_ndarray((numpy.dot(self.A, points.T) >= self.b[:,None]).all(axis=0))
        elif points.ndim == 1:
            return ge_polyhedron.ineqs_satisfied(self, numpy.array([points]))[0]

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

    
    def column_bounds(self) -> numpy.array:

        """
            Returns the initial bounds for each variable.

            Returns
            -------
                out : numpy.ndarray
        """
        return numpy.array(
            list(
                map(
                    maz.compose(
                        operator.methodcaller("as_tuple"),
                        operator.attrgetter("bounds"),
                    ),
                    self.A.variables
                )
            )
        ).T

    @property
    def n_row_combinations(self) -> int:

        """
            The number of combinations row wise excluding its constraint.

            Examples
            --------
                >>> puan.ndarray.ge_polyhedron([[1,1,1,0,0],[1,1,0,0,0]]).n_row_combinations
                array([4, 2])

                >>> puan.ndarray.ge_polyhedron([[3,4,5,6,0],[7,6,5,4,3]]).n_row_combinations
                array([ 8, 16])

                >>> puan.ndarray.ge_polyhedron([[3,4,5]], variables=[puan.variable("0"), puan.variable("a", (-2,2)), puan.variable("b", (-10,10))]).n_row_combinations
                array([105])
        """
        bnds = self.column_bounds()
        return numpy.array(numpy.prod((self.A != 0)*(bnds[1]-bnds[0]+1) + (self.A == 0)*1, axis=1))

    def tighten_column_bounds(self) -> numpy.array:
        
        """
            Returns maybe tighter column/variable bounds based on row constraints.

            Notes
            -----
            - Array returned has two rows - first is the lower bound and the second is the upper bound, on each column.
            - Lower bound may be larger than upper bound. This implies contradiction but no exception is raised here.

            Examples
            --------
                >>> ge_polyhedron([[0,-2,1,1,0],[1,1,0,0,0],[1,0,0,0,1],[-3,0,0,0,-1]]).tighten_column_bounds()
                array([[1, 0, 0, 1],
                       [1, 1, 1, 1]])

                >>> ge_polyhedron([[3,1,1,1,0]]).tighten_column_bounds()
                array([[1, 1, 1, 0],
                       [1, 1, 1, 1]])

                >>> ge_polyhedron([[0,-1,-1,0,0]]).tighten_column_bounds()
                array([[0, 0, 0, 0],
                       [0, 0, 1, 1]])

                >>> ge_polyhedron([[2,-1,-1,0]], variables=[puan.variable(0, dtype="int"), puan.variable("a", bounds=(-2,1)), puan.variable("b"), puan.variable("c")]).tighten_column_bounds()
                array([[-2,  0,  0],
                       [-2,  0,  1]])

            Returns
            -------
                out : numpy.array
        """
        cm_bnds = self.column_bounds()
        min_value = puan.default_min_int
        max_value = puan.default_max_int
        with numpy.errstate(all='ignore'):
            rw_bnds = self.row_bounds()
            res = numpy.floor(-(rw_bnds.T[1].reshape(-1,1) - self.A_max) / self.A)
            res[res == numpy.inf] = max_value
            res[res == -numpy.inf] = min_value
            lbs = res.copy()
            ubs = res.copy()
            lbs[self.A <= 0] = min_value
            ubs[self.A >= 0] = max_value
            lb_mx = numpy.max(lbs, axis=0)
            ub_mn = numpy.min(ubs, axis=0)
            lb_msk = lb_mx > cm_bnds[0]
            ub_msk = ub_mn < cm_bnds[1]
            cm_bnds[0, lb_msk] = lb_mx[lb_msk]
            cm_bnds[1, ub_msk] = ub_mn[ub_msk]

        return cm_bnds

    def row_bounds(self) -> numpy.ndarray:

        """
            Returns the row equation bounds, inclusive the bias.
            For instance, x+y+z>=1 has bounds of (-1,2).

            Examples
            --------
            >>> ge_polyhedron(numpy.array([[-2, -3,  0,  0,  0]])).row_bounds()
            array([[-1,  2]])

            Returns
            -------
                out : numpy.ndarray
        """
        clm_bounds = self.column_bounds()
        A_ = numpy.array([
            clm_bounds[0]*self.A,
            clm_bounds[1]*self.A,
        ])
        return numpy.array([
            A_.min(axis=0).sum(axis=1)-self.b, 
            A_.max(axis=0).sum(axis=1)-self.b
        ]).T

    def row_distribution(self, row_index: int) -> numpy.ndarray:

        """
            Returns a distribution of all combinations for each row and their values.
            Data type is a 2D numpy array with two columns: 
            Column index 0 is a range from lowest to highest value of row equation. 
            Column index 1 is a counter of how many combinations, generated from variable bounds, evaluated into that value.

        
            .. code-block::
                :caption: Example 1
                
                Equation = x+y+0
                Result   = array([
                    [0, 1],
                    [1, 2],
                    [2, 1]
                ])

                2     |         <- 2 combinations ([1,0], [0,1]) results in value 1
                1  |  |  |      <- 1 combination  ([0,0]) results in value 0
                  ----------       and 1 combination ([1,1]) results in value 2
                   0  1  2   

            .. code-block::
                :caption: Example 2

                Equation = 2x+2y+0
                Result   = array([
                    [0, 1],
                    [1, 0],
                    [2, 2],
                    [3, 0],
                    [4, 1]
                ])

                2        |   
                1  |     |     |
                  ---------------
                   0  1  2  3  4

            Notes
            -----
            This operation will generate all combinations from column bounds and may require heavy computation. 
            It is supposed to be used as an analysis tool. Use it with caution.

            Examples
            --------
                >>> ge_polyhedron([[0,1,1,0],[0,2,2,0],[0,3,3,3]]).row_distribution(0)
                array([[0, 1],
                       [1, 2],
                       [2, 1]])

                >>> ge_polyhedron([[0,1,1,0],[0,2,2,0],[0,3,3,3]]).row_distribution(1)
                array([[0, 1],
                       [1, 0],
                       [2, 2],
                       [3, 0],
                       [4, 1]])

                >>> ge_polyhedron([[0,1,1,0],[0,2,2,0],[0,3,3,3]]).row_distribution(2)
                array([[0, 1],
                       [1, 0],
                       [2, 0],
                       [3, 3],
                       [4, 0],
                       [5, 0],
                       [6, 3],
                       [7, 0],
                       [8, 0],
                       [9, 1]])

            See also
            --------
                row_stretch      : Proportion of the number of value spots for each row equation.
                row_stretch_int  : Row bounds range from one number to another.

            Raises
            ------
                Exception
                    | Row index out of bounds.
                    | If variable mismatch between polyhedron variables and equation.

            Returns
            -------
                out : numpy.ndarray
        """
        if not (row_index >= 0 and row_index < self.shape[0]):
            raise Exception(f"row index {row_index} is out of bounds in polyhedron of shape {self.shape}")

        variable_bounds_full = numpy.array(
            list(
                map(
                    lambda x: list(range(x[0], x[1]+1)),
                    self.column_bounds().T
                )
            )
        )
        eq = self[row_index]
        msk_a = eq[1:] != 0
        if not len(eq)-1 == len(variable_bounds_full):
            raise Exception(f"number of variables mismatch in equation and bounds")
        
        sorted_counter = list(
            Counter(
                map(
                    lambda combination: eq[1:][msk_a].dot(combination) - eq[0],
                    itertools.product(*variable_bounds_full[msk_a])
                )
            ).items()
        )
        keys,_ = list(zip(*sorted_counter))
        keys, values = zip(
            *sorted(
                sorted_counter + list(
                    map(
                        lambda x: (x, 0),
                        set(range(min(keys), max(keys)+1)).difference(keys)
                    )
                )
            )
        )

        return numpy.array([list(keys), list(values)]).T

    def row_stretch(self) -> numpy.array:

        """
            This shows the proportion of the number of value spots for each
            row equation with respect to the number of combinations from active variable bounds. 
            It will return a vector of equal size as number of rows in polyhedron. Each value
            less than 1 means that there are more value spots than the number 
            of possible combinations for that row.

            See also
            --------
                row_distribution : Distribution of possible equation outcomes.
                row_stretch_int  : Row bounds range from one number to another.

            Examples
            --------
                >>> ge_polyhedron([[1,1,1,0,0],[2,2,2,0,0]]).row_stretch()
                integer_ndarray([1.33333333, 0.8       ])

            Returns
            -------
                out : numpy.array
        """
        cm_bnds = self.column_bounds()
        rw_bnds = self.row_bounds()
        return numpy.prod((cm_bnds[1]-cm_bnds[0])*(self.A != 0)+1, axis=1) / (rw_bnds[:,1]-rw_bnds[:,0]+1)

    def row_stretch_int(self, row_index: int) -> numpy.ndarray:

        """
            Row bounds range from one number to another. They normally
            have at least one combination for each number in this range.
            This function returns a number <=0 for each row bound representing
            its "stretch". If the number is <0, then there exists a gap in
            that row's bounds.

            As in example 2, there are no valid combinations summing up to 2 or 4. What this
            means is that the equation has unnecessary large coefficients and may have negative
            effects on other methods assuming a certain underlying equation structure.

            See also
            --------
                row_distribution : Distribution of possible equation outcomes.
                row_stretch      : Proportion of the number of value spots for each row equation.

            Examples
            --------
                >>> ph = ge_polyhedron([[0,1,1],[0,2,2]])
                >>> ph.row_stretch_int(0)
                0

                >>> ph = ge_polyhedron([[0,1,1],[0,2,2]])
                >>> ph.row_stretch_int(1)
                -2

            Notes
            -----
            This operation uses `row_distribution` which will generate all combinations from variable bounds and may require heavy computation. 
            It is supposed to be used as an analysis tool. Use it with caution.

            Returns
            -------
                out : numpy.ndarray
        """
        rng, n = self.row_distribution(row_index).T
        return int((n != 0).sum() - (rng[-1]-rng[0]) -1)


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
        nvars = self.shape[self.ndim-1]
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

    def to_list(self, skip_virtual_variables: bool = False) -> typing.List[puan.variable]:

        """
            Returns a list of variables for each value in self equal 1.

            Returns
            -------
                out : list : puan.variable
        """

        if self.ndim == 1:
            return numpy.array(self.variables)[self == 1].tolist()
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
        nvars = self.shape[self.ndim-1]
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


class InfeasibleError(Exception):
    """In context of solving on polyhedron and no feasible solution exists"""
    pass

class ge_polyhedron_config(ge_polyhedron):

    """A ``puan.ge_polyhedron`` sub class with specific configurator features."""

    def __new__(cls, input_array, default_prio_vector: numpy.ndarray = None, variables: typing.List[puan.variable] = [], index: typing.List[typing.Union[int, puan.variable]] = [], dtype=numpy.int64):
        arr = super().__new__(cls, input_array, variables=variables, index=index, dtype=dtype)
        arr.default_prio_vector = default_prio_vector if default_prio_vector is not None else -numpy.ones((arr.A.shape[1]))
        return arr

    def _vectors_from_prios(self, prios: typing.List[typing.Dict[str, int]]) -> numpy.ndarray:

        """
            Constructs weight vectors from prioritization list(s).
        """

        return integer_ndarray(
            numpy.array(
                list(
                    map(
                        lambda y: [
                            self.default_prio_vector,
                            list(
                                map(
                                    maz.compose(
                                        maz.pospartial(
                                            dict.get,
                                            [
                                                (0, y),
                                                (2, 0)
                                            ]
                                        ),
                                        operator.attrgetter("id")
                                    ),
                                    self.A.variables
                                )
                            )
                        ],
                        prios
                    )
                )
            )
        ).ndint_compress(method="shadow", axis=0)

    def select(
        self, 
        *prios: typing.List[typing.Dict[str, int]], 
        solver: typing.Callable[[ge_polyhedron, typing.Dict[str, int]], typing.Iterable[typing.Tuple[typing.List[int], int, int]]] = None,
    ) -> typing.List[typing.List[puan.SolutionVariable]]:

        """
            Select items to prioritize and receives a solution for each in prio's list.

            Parameters
            ----------
                *prios : typing.List[typing.Dict[str, int]]
                    a list of dicts where each entry's value is a prio

                solver: typing.Callable[[ge_polyhedron, typing.Dict[str, int]], typing.List[(np.ndarray, int, int)]] = None
                    If None is provided puan's own (beta) solver is used. If you want to provide another solver
                    you have to send a function as solver parameter. That function has to take a `ge_polyhedron` and
                    a 2D numpy array representing all objectives, as input. NOTE that the polyhedron DOES NOT provide constraints for variable
                    bounds. Variable bounds are found under each variable under `polyhedron.variables` and constraints for 
                    these has to manually be created and added to the polyhedron matrix. The function should return a list, one for each
                    objective, of tuples of (solution vector, objective value, status code). The solution vector is an integer ndarray vector
                    of size equal to width of `polyhedron.A`. There are six different status codes from 1-6:
                        - 1: solution is undefined
                        - 2: solution is feasible
                        - 3: solution is infeasible
                        - 4: no feasible solution exists
                        - 5: solution is optimal
                        - 6: solution is unbounded

                    Checkout https://github.com/ourstudio-se/puan-solvers for quick how-to's for common solvers.

            Examples
            --------
                >>> ph = ge_polyhedron_config([[1,1,1,1,0],[-1,-1,-1,-1,0]])
                >>> ph.select({"1": 1})[0]
                [SolutionVariable(id=1, bounds=Bounds(lower=0, upper=1))]

                >>> ph = ge_polyhedron_config([[1,1,1,1,0],[-1,-1,-1,-1,0]])
                >>> dummy_solver = lambda x, y: list(map(lambda v: (v, 0, 5), y))
                >>> ph.select({"1": 1}, solver=dummy_solver)[0]
                [SolutionVariable(id=1, bounds=Bounds(lower=0, upper=1)), SolutionVariable(id=2, bounds=Bounds(lower=0, upper=1)), SolutionVariable(id=3, bounds=Bounds(lower=0, upper=1)), SolutionVariable(id=4, bounds=Bounds(lower=0, upper=1))]

            Raises
            ------
            InfeasibleError
                No solution could be found. Note that if solver raises another
                error, it will be shown within parantheses.

            Returns
            -------
                out : typing.List[typing.List[puan.SolutionVariable]]
        """
        try:
            variables = self.A.variables
            objectives = self._vectors_from_prios(prios)
            id_map = dict(
                zip(
                    range(self.A.shape[1]), 
                    variables
                )
            )
            if solver is None:
                id_map_rev = dict(zip(map(lambda x: x.id, id_map.values()), id_map.keys()))
                solutions = list(
                    map(
                        lambda int_sol: (
                            int_sol.x, 
                            int_sol.z, 
                            int_sol.status_code,
                        ),
                        prs.PolyhedronPy(
                            prs.MatrixPy(
                                self.A.flatten().tolist(),
                                *self.A.shape
                            ),
                            self.b.tolist(),
                            list(
                                map(
                                    lambda variable: prs.VariableFloatPy(
                                        id_map_rev[variable.id],
                                        (
                                            float(variable.bounds.lower),
                                            float(variable.bounds.upper),
                                        ),
                                    ),
                                    variables
                                )
                            ),
                            list(range(self.A.shape[0])),
                        ).solve(
                            list(
                                map(
                                    lambda v: dict(
                                        zip(
                                            id_map.keys(), 
                                            v
                                        )
                                    ),
                                    objectives,
                                )
                            )
                        )
                    )
                )
            else:
                solutions = solver(
                    self,
                    objectives,
                )

            return list(
                itertools.starmap(
                    lambda x, _, status_code: list(
                        itertools.starmap(
                            lambda i,v: puan.SolutionVariable.from_variable(
                                id_map[i],
                                v
                            ), 
                            filter(
                                lambda y: y[1] != 0,
                                zip(
                                    id_map.keys(),
                                    x,
                                )
                            )
                        )
                    ) if status_code in [5,6] else None,
                    solutions,
                )
            )
        except Exception as e:
            raise InfeasibleError(f"couldn't generate a solution from solver ({e})")

    def to_b64(self, str_decoding: str = 'utf8') -> str:

        """
            Packs data into a base64 string.

            Parameters
            ----------
                str_decoding: str = 'utf8'

            Returns
            -------
                out : str
        """
        return base64.b64encode(
            gzip.compress(
                pickle.dumps(
                    [self, self.default_prio_vector, self.variables, self.index, self.dtype],
                    protocol=pickle.HIGHEST_PROTOCOL,
                ),
                mtime=0,
            )
        ).decode(str_decoding)

    @staticmethod
    def from_b64(base64_str: str) -> "ge_polyhedron_config":

        """
            Unpacks base64 string `base64_str` into some data.

            Parameters
            ----------
                base64_str: str

            Raises
            ------
                Exception
                    When error occurred during decompression.

            Returns
            -------
                out : dict
        """
        try:
            return ge_polyhedron_config(
                *pickle.loads(
                    gzip.decompress(
                        base64.b64decode(
                            base64_str.encode()
                        )
                    )
                )
            )
        except:
            raise Exception("could not decompress and load polyhedron from string: version mismatch.")


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
