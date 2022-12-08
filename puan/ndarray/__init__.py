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
import puan_rspy as pr
from enum import IntEnum

from collections import Counter

class variable_ndarray(numpy.ndarray):
    """
        A :class:`numpy.ndarray` sub class which ties variables to the indices of the :class:`numpy.ndarray`.

        Attributes
        ----------
            See : :class:`numpy.ndarray`
        
        Raises
        ------
            ValueError
                If shapes between ``input_array``, ``variables`` and ``index mismatch``

        Methods
        -------
        variable_indices
        boolean_variable_indices
        integer_variable_indices
        construct
    """
    def __new__(cls, input_array: numpy.ndarray, variables: typing.List[puan.variable] = [], index: typing.List[typing.Union[int, puan.variable]] = [], dtype: typing.Type=numpy.int64):
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
    def _default_variable_list(n, default_bounds_type: str = "bool") -> typing.List[puan.variable]:

        """
            Generates a default list of variables with first variable
            as the support vector variable default from :class:`puan.variable`.

            Returns
            -------
                out : List[:class:`puan.variable`]
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

    def variable_indices(self, variable_dtype: puan.Dtype) -> numpy.ndarray:

        """
            Variable indices of variable type :class:`puan.Dtype.BOOL` or :class:`puan.Dtype.INT`.

            Parameters
            ----------
            variable_dtype : :class:`puan.Dtype`
                Variable type where "bool" gives :class:``puan.Dtype.BOOL`` and 1 gives :class:``puan.Dtype.INT``.
            
            Raises
            ------
                ValueError
                    If variable_dtype is not of type :class:`puan.Dtype`

            Notes
            -----
                Variables of bounds other than (0,1) are considered of type :class:`puan.Dtype.INT`.

            Returns
            -------
                out : :class:`numpy.ndarray`
        """
        
        is_bool = 1*(variable_dtype==puan.Dtype.BOOL)
        is_int = 1*(variable_dtype==puan.Dtype.INT)
        if is_bool + is_int != 1:
            raise ValueError("Unrecognized variable type, must be `puan.Dtype` got {}".format(variable_dtype))
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
            Variable indices where variable dtype is :class:`puan.Dtype.BOOL`.

            Returns
            -------
                out : :class:`numpy.ndarray`

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

        return self.variable_indices(puan.Dtype.BOOL)

    @property
    def integer_variable_indices(self) -> numpy.ndarray:

        """
            Variable indices where variable dtype is :class:`puan.Dtype.INT`.

            Returns
            -------
                out : :class:`numpy.ndarray`

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

        return self.variable_indices(puan.Dtype.INT)

    def construct(self, variable_values: typing.Dict[str, int], default_value: typing.Optional[typing.Callable[["puan.variable"], typing.Union[int, float]]] = None, dtype: typing.Type = numpy.int64) -> numpy.ndarray:

        """
            Constructs a :class:`variable_ndarray` from a dict of variable ids and integers.

            Parameters
            ----------
                variable_values: Dict[str, int]
                    List of tuples with variable id and value
                default_value : Callable
                    function taking self and returning a :class:`numpy.ndarray` with default values with shape equal to the number of variables.
                    Default is 0 when dtype is :class:`numpy.int64` and ``numpy.nan`` otherwise.
                dtype : Type
                    Type of resulting :class:`numpy.ndarray`. Default is :class:`numpy.int64`.

            Examples
            --------

            Constructing a new 1d variable ndarray shadow from this array and setting ``x = 5``
                >>> vnd = variable_ndarray([[1,2,3], [2,3,4]], [puan.variable("x"), puan.variable("y"), puan.variable("z")])
                >>> vnd.construct({"x": 5})
                array([5, 0, 0])

            Notes
            -----
            If a variable in ``variable_values`` is not in ``self.variables``, it will be ignored.

            Returns
            -------
                out : :class:`numpy.ndarray`
        """
        return numpy.array(
            list(
                map(
                    maz.ifttt(

                        # If variable is in variable_values, 
                        maz.compose(
                            functools.partial(
                                operator.contains,
                                variable_values
                            ),
                            operator.attrgetter("id"),
                        ),

                        # then get the value
                        maz.compose(
                            variable_values.get,
                            operator.attrgetter("id"),
                        ),

                        # else, pick a default value depending
                        # on default_value function and dtype
                        maz.ifttt(
                            # if default value is a function
                            lambda _: callable(default_value),

                            # then let that function decide the variable value
                            default_value,

                            # else pick a value based dtype
                            maz.ifttt(
                                
                                # if dtype is inherited from int,
                                lambda _: issubclass(dtype, (int, numpy.integer)),
                                
                                # then select the variable's lower bound.
                                operator.attrgetter("bounds.lower"),

                                # else, choose np.nan
                                lambda _: numpy.nan,
                            )
                        ),
                    ),
                    self.variables,
                )
            ),
            dtype=dtype,
        )

class ge_polyhedron(variable_ndarray):
    """
        A :class:`numpy.ndarray` sub class and a system of linear inequalities forming
        a polyhedron. The "ge" stands for "greater or equal" (:math:`\\ge`)
        which represents the relation between :math:`A` and :math:`b` (as in :math:`Ax \\ge b`), i.e.
        polyhedron :math:`P=\{x \\in R^n \ |\  Ax \\ge b\}`.

        Attributes
        ----------
            See : :class:`numpy.ndarray`

        Methods
        -------
        to_linalg
        reducable_columns_approx
        reduce_columns
        reducable_rows
        reduce_rows
        reducable_rows_and_columns
        reduce
        row_bounds
        row_distribution
        row_stretch
        row_stretch_int
        neglectable_columns
        neglect_columns
        separable
        ineq_separate_points

    """

    def __new__(cls, input_array: numpy.ndarray, variables: typing.List[puan.variable] = [], index: typing.List[typing.Union[int, puan.variable]] = [], dtype: typing.Type=numpy.int64):
        return super().__new__(cls, input_array, variables=variables, index=index, dtype=dtype)

    @property
    def A(self) -> "integer_ndarray":

        """
            Matrix :math:`A`, as in :math:`Ax \\ge b`.

            Returns
            -------
                out : :class:`integer_ndarray`

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
    def A_max(self) -> "integer_ndarray":

        """
            Returns the maximum coefficient value based on variable's initial bounds.

            Examples
            --------
            >>> ge_polyhedron(numpy.array([
            ...     [0,-1, 1, 0, 0],
            ...     [0, 0,-1, 1, 0],
            ...     [0, 0, 0,-1, 1]])).A_max
            integer_ndarray([[0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

            Raises
            ------
                Exception
                    If matrix A is empty.

            Returns
            -------
                out : :class:`integer_ndarray`
        """
        init = self.column_bounds()
        if init.size == 0:
            raise Exception(f"Matrix A in polyhedron is empty")
        return (init[0]*(self.A<0)*self.A)+(init[1]*(self.A>0)*self.A)

    @property
    def A_min(self) -> "integer_ndarray":

        """
            Returns the minimum coefficient value based on variable's initial bounds.

            Examples
            --------
            >>> ge_polyhedron(numpy.array([
            ...     [0,-1, 1, 0, 0],
            ...     [0, 0,-1, 1, 0],
            ...     [0, 0, 0,-1, 1]])).A_min
            integer_ndarray([[-1,  0,  0,  0],
                             [ 0, -1,  0,  0],
                             [ 0,  0, -1,  0]])

            Raises
            ------
                Exception
                    If matrix A is empty.

            Returns
            -------
                out : :class:`integer_ndarray`
        """
        init = self.column_bounds()
        if init.size == 0:
            raise Exception(f"Matrix A in polyhedron is empty")

        return (init[0]*(self.A>0)*self.A)+(init[1]*(self.A<0)*self.A)

    @property
    def b(self) -> "integer_ndarray":

        """
            Support vector :math:`b`, as in :math:`Ax \\ge b`.

            Returns
            -------
                out : :class:`integer_ndarray`

            Examples
            --------
                >>> ge_polyhedron(numpy.array([
                ...     [0,-1, 1, 0, 0],
                ...     [0, 0,-1, 1, 0],
                ...     [0, 0, 0,-1, 1]])).b
                integer_ndarray([0, 0, 0])
        """
        return integer_ndarray(self.T[0])

    def to_linalg(self) -> typing.Tuple["integer_ndarray", numpy.ndarray]:
        """
            Assumes support vector index 0 in polyhedron
            and returns :math:`A, b` as in :math:`Ax \\ge b`

            Returns
            -------
                out : Tuple[:class:`integer_ndarray`, :class:`numpy.ndarray`]
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
                                 [ 0,  0, -1,  1]]), integer_ndarray([0, 0, 0]))
        """
        return self.A, self.b

    def reducable_columns_approx(self) -> numpy.ndarray:
        """
            Returns which columns are reducable under approximate condition.
            The approximate condition is that only one row of :class:`ge_polyhedron` is
            considered when deducing reducable columns. By considering combination of rows
            more reducable columns might be found.

            Rows with values :math:`\\neq0` for any integer variable are neglected.

            Returns
            -------
                out : :class:`numpy.ndarray` (vector)
                    Columns with positive values could be assumed.
                    Columns with nonpositive values could be removed (not-assumed).

            See also
            --------
                reduce : Reduces matrix polyhedron by information passed in ``rows_vector`` and ``columns_vector``.
                reduce_columns : Reducing columns from ``polyhedron`` from ``columns_vector`` where a positive number meaning *assume* and a nonpositive number meaning *not assume*.
                reducable_rows_and_columns : Returns reducable rows and columns of given polyhedron.
                reduce_rows : Reduces rows from ``rows_vector`` where num of rows of ``M`` equals size of ``rows_vector``.
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

    def reduce_columns(self, columns_vector: numpy.ndarray) -> "ge_polyhedron":

        """
            Reducing columns from polyhedron from ``columns_vector`` where a positive number meaning *assume*
            and a nonpositive number meaning *not assume*.

            Parameters
            ----------
            columns_vector : :class:`numpy.ndarray`
                The polyhedron is reduced column-wise by equally many positives and nonpositive in ``columns_vector``.

            Returns
            -------
                out : :class:`ge_polyhedron`

            See also
            --------
                reduce : Reduces matrix polyhedron by information passed in ``rows_vector`` and ``columns_vector``.
                reducable_rows_and_columns : Returns reducable rows and columns of given polyhedron.
                reduce_rows : Reduces rows from ``rows_vector`` where num of rows of ``M`` equals size of ``rows_vector``.
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

    def reducable_rows(self) -> "boolean_ndarray":
        """
            Returns a boolean vector indicating what rows are reducable.
            A row is reducable iff it doesn't constrain any variable.

            Rows with values :math:`\\neq0` for any integer variable are neglected.

            Returns
            -------
                out : :class:`integer_ndarray` (vector)

            See also
            --------
                reduce : Reduces matrix polyhedron by information passed in ``rows_vector`` and ``columns_vector``.
                reduce_columns : Reducing columns from ``polyhedron`` from ``columns_vector`` where a positive number meaning *assume* and a nonpositive number meaning *assume*.
                reducable_rows_and_columns : Returns reducable rows and columns of given polyhedron.
                reduce_rows : Reduces rows from ``rows_vector`` where num of rows of ``M`` equals size of ``rows_vector``.
                reducable_columns_approx : Returns what columns are reducable under approximate condition.

            Examples
            --------
            The sum of all negative numbers of the row in :math:`A` is :math:`\\ge b`, i.e.
            :math:`Ax \\ge b` will always hold, regardless of :math:`x`.

                >>> ge_polyhedron(numpy.array([[-3, -1, -1, 1, 0]])).reducable_rows()
                boolean_ndarray([1])

            All elements of the row in :math:`A` is :math:`\\ge 0` and :math:`b` is :math:`\\le 0`,
            again :math:`Ax \\ge b` will always hold, regardless of :math:`x`.

                >>> ge_polyhedron(numpy.array([[0, 1, 1, 1, 0]])).reducable_rows()
                boolean_ndarray([1])

        """
        return boolean_ndarray(self.A_min.sum(axis=1) >= self.b)

    def reduce_rows(self, rows_vector: "boolean_ndarray") -> "ge_polyhedron":

        """
            Reduces rows from a ``rows_vector`` where num of rows of :class:`ge_polyhedron` equals
            size of ``rows_vector``. Each row in ``rows_vector`` equal to 0 is kept.

            Parameters
            ----------
                rows_vector : :class:`boolean_ndarray`

            Returns
            -------
                out : :class:`ge_polyhedron`

            See also
            --------
                reduce : Reduces matrix polyhedron by information passed in ``rows_vector`` and ``columns_vector``.
                reduce_columns : Reducing columns from ``polyhedron`` from ``columns_vector`` where a positive number meaning *assume* and a nonpositive number meaning *assume*.
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

    def reducable_rows_and_columns(self) -> typing.Tuple["boolean_ndarray", numpy.ndarray]:

        """
            Returns reducable rows and columns of given polyhedron.

            Returns
            -------
                out : Tuple[:class:`boolean_ndarray`, :class:`numpy.ndarray`]
                    out[0] : a vector equal size as ``ge_polyhedron``'s row size where 1 represents a removed row and 0 represents a kept row\n
                    out[1] : a vector with equal size as ``ge_polyhedron``'s column size where nan numbers represent a no-choice.

            See also
            --------
                reduce : Reduces matrix polyhedron by information passed in ``rows_vector`` and ``columns_vector``.
                reduce_rows : Reduces rows from ``rows_vector`` where num of rows of ``M`` equals size of ``rows_vector``.
                reducable_rows : Returns a boolean vector indicating what rows are reducable.
                reduce_columns :  Reducing columns from ``polyhedron`` from ``columns_vector`` where a positive number meaning *assume* and a nonpositive number meaning *assume*.
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
                (boolean_ndarray([0, 0, 0, 1, 1, 1, 1]), array([nan, nan, nan,  1.,  0.]))

        """

        _M = self.copy()
        red_cols = ge_polyhedron.reducable_columns_approx(_M)
        red_rows = ge_polyhedron.reducable_rows(_M) * 1
        full_cols = numpy.zeros(_M.A.shape[1], dtype=int)*numpy.nan
        full_rows = boolean_ndarray(numpy.zeros(_M.shape[0], dtype=int))
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

    def reduce(self, rows_vector: "boolean_ndarray"=None, columns_vector: numpy.ndarray=None) -> "ge_polyhedron":
        """
            Reduces matrix polyhedron by information passed in ``rows_vector`` and ``columns_vector``.

            Parameters
            ----------
            rows_vector : :class:`boolean_ndarray` (optional)
                A vector of 0's and 1's where rows matching index of value 1 are removed.

            columns_vector : :class:`numpy.ndarray` (optional)
                A vector of positive, nonpositive and ``NaN floats``. The ``NaN``'s represent a no-choice.

            Returns
            -------
                out : :class:`ge_polyhedron`
                    Reduced polyhedron

            See also
            --------
                reducable_rows_and_columns : Returns reducable rows and columns of given polyhedron.
                reduce_rows : Reduces rows from ``rows_vector`` where num of rows of ``M`` equals size of ``rows_vector``.
                reducable_rows : Returns a boolean vector indicating what rows are reducable.
                reduce_columns :  Reducing columns from ``polyhedron`` from ``columns_vector`` where a positive number meaning *assume* and a nonpositive number meaning *assume*.
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

    def neglectable_columns(self, patterns: numpy.ndarray) -> "integer_ndarray":
        """
            Returns neglectable columns of given polyhedron :class:`ge_polyhedron` based on given patterns.
            Neglectable columns are the columns which doesn't differentiate the patterns in :class:`ge_polyhedron`
            from the patterns not in :class:`ge_polyhedron`

            Parameters
            ----------
                patterns : :class:`numpy.ndarray`
                    known patterns of variables in the polyhedron.

            Returns
            -------
                out : :class:`integer_ndarray`
                    A 1d ndarray of length equal to the number of columns of the ge_polyhedron,
                    with ones at corresponding columns which can be neglected, zero otherwise.

            See also
            --------
                :meth:`neglect_columns` : Neglects columns in :math:`A` from ``columns_vector``.

            Notes
            -----
                This method is differentiating the patterns which are in :class:`ge_polyhedron` from those that are not.
                Variables which are not in the patterns and has a positive number for any row in :class:`ge_polyhedron`
                are considered non-neglectable.

            Examples
            --------
            Keep common pattern:
            :class:`ge_polyhedron` with two out of three patterns. Variables 1 and 2 are not differentiating the patterns
            in :class:`ge_polyhedron` from those that are not, and can therefore be neglected.

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

            All patterns are present and all those columns can therefore be neglected
            >>> patterns = numpy.array([[1, 1, 0], [1, 0, 1], [1, 0, 0]])
            >>> ge_polyhedron(numpy.array([
            ...     [-1,-1,-1, 0, 0, 0, 1],
            ...     [-1,-1, 0,-1, 0, 0, 1],
            ...     [-1,-1, 0, 0, 0, 0, 1]])).neglectable_columns(patterns)
            integer_ndarray([1, 1, 1, 0, 0, 0])

            

        """
        A = ge_polyhedron(self).A
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
            return integer_ndarray(~non_neglectable_columns)

        # Find common pattern in A
        common_pattern_in_A = (_A == 1).all(axis=0)
        if not (patterns_not_in_A[:,common_pattern_in_A]==1).all(axis=1).any(axis=0):
            # Possible to neglect everything except the common pattern and the non neglectable columns
            return (~(common_pattern_in_A | non_neglectable_columns)).astype(int)
        return ((_A[:, (patterns_not_in_A==0).all(axis=0) & (non_neglectable_columns==0)]).any(axis=1).all()) * (patterns_not_in_A!=0).any(axis=0).astype(int)

    def neglect_columns(self, columns_vector: numpy.ndarray) -> "ge_polyhedron":
        """
            Neglects columns in :math:`A` from ``columns_vector``. The entire column for col in ``columns_vector`` :math:`>0`
            is set to :math:`0` and the support vector is updated.

            Parameters
            ----------
                columns_vector : :class:`numpy.ndarray`
                    A 1d ndarray of length equal to the number of columns of :math:`A`.
                    Sets the entire column of :math:`A` to zero for corresponding entries of ``columns_vector`` which are :math:`>0`.

            Returns
            -------
                out : :class:`ge_polyhedron`
                    :class:`ge_polyhedron` with neglected columns set to zero and support vector updated accordingly.

            See also
            --------
                :meth:`neglectable_columns` : Returns neglectable columns of given polyhedron :class:`ge_polyhedron` based on given patterns.

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

    def separable(self, points: numpy.ndarray) -> numpy.ndarray:

        """
            Checks if points are separable by a hyperplane from the polyhedron.

            .. code-block::

                           / \\
                          /   \\
                point -> /  x  \\
                        /_ _ _ _\ <- polyhedron

            Parameters
            ----------
                points : :class:`numpy.ndarray`
                    Points to be evaluated if separable to polyhedron by a hyperplane.

            Returns
            -------
                out : :class:`numpy.ndarray`
                    Boolean vector indicating T if separable to polyhedron by a hyperplane.

            See also
            --------
                :meth:`ineq_separates_points` : Checks if a linear inequality of the polyhedron separates any point of given points.

            Examples
            --------
                >>> polyhedron = ge_polyhedron([[1, 1, 1], [-1, -1, -1]])
                >>> polyhedron.separable(numpy.array([0, 0]))
                True

                >>> polyhedron = ge_polyhedron([[0,-2, 1, 1]])
                >>> polyhedron.separable(numpy.array([
                ...     [1, 0, 1],
                ...     [1, 1, 1],
                ...     [0, 0, 0]]))
                array([ True, False, False])

                >>> polyhedron = ge_polyhedron([[-2,-1, -1, -1], [0, -1, 1, 0], [0, 0, -1, 1]])
                >>> polyhedron.separable(numpy.array([[
                ...     [1, 0, 1]],
                ...     [[1, 1, 1]],
                ...     [[0, 0, 0]]]))
                array([[ True],
                       [ True],
                       [False]])
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

    def ineq_separate_points(self, points: numpy.ndarray) -> "boolean_ndarray":

        """
            Checks if a linear inequality of the polyhedron separates any point from the polyhedron.

            One linear equalities separates one of the given points here::


                           -/ \\-
                          -/   \-  x <- point
                point -> -/  x  \\-
                        -/_ _ _ _\- <- polyhedron
                          / / / /

            Parameters
            ----------
                points : :class:`numpy.ndarray`

            Returns
            -------
                out : :class:`boolean_ndarray`
                    boolean vector indicating if linear inequality enfolds all points

            See also
            --------
                :meth:`separable` : Checks if points are inside the polyhedron.

            Notes
            -----
                This function is the inverse of :meth:`separable`

            Examples
            --------
            Points in 1d

            >>> points = numpy.array([0, 1])
            >>> ge_polyhedron(numpy.array([
            ...     [-1,  0, -1],
            ...     [ 2,  1,  1],
            ...     [ 0, -1,  1]])).ineq_separate_points(points)
            boolean_ndarray([0, 1, 0])

            Points in 2d

            >>> points = numpy.array([[1, 1], [4, 2]])
            >>> ge_polyhedron(numpy.array([
            ...     [ 0, 1, 0],
            ...     [ 0, 1,-1],
            ...     [-1,-1, 1]])).ineq_separate_points(points)
            boolean_ndarray([0, 0, 1])

            Points in 3d

            >>> points = numpy.array([
            ...     [[1, 1, 1],
            ...      [4, 2, 1]],
            ...     [[0, 1, 0],
            ...      [1, 2, 1]]])
            >>> ge_polyhedron(numpy.array([
            ...     [ 0, 1, 0,-1],
            ...     [ 0, 1,-1, 0],
            ...     [-1,-1, 1,-1]])).ineq_separate_points(points)
            boolean_ndarray([[0, 0, 1],
                             [0, 1, 0]])

        """
        if points.ndim > 2:
            return boolean_ndarray(
                list(
                    map(self.ineq_separate_points, points)
                )
            )
        elif points.ndim == 2:
            A, b = self.to_linalg()
            return boolean_ndarray(
                (numpy.matmul(A, points.T) < b.reshape(-1,1)).any(axis=1)
            )
        elif points.ndim == 1:
            return ge_polyhedron.ineq_separate_points(self, numpy.array([points]))
    
    def ineqs_satisfied(self, points: numpy.ndarray) -> "boolean_ndarray":
        """
            Checks if the linear inequalities in the polyhedron are satisfied given the points, i.e. :math:`Ap \\ge b`, given :math:`p` in points.

            Parameters
            ----------
                points : :class:`numpy.ndarray`

            Returns
            -------
                out : :class:`boolean_ndarray`
                    boolean vector indicating if the linear inequality is satisfied given all points

            Examples
            --------

            Points in 1d

            >>> points = numpy.array([1, 0, 3, 2])
            >>> ge_polyhedron(numpy.array([
            ...     [0, 1, 1, -1, 1],
            ...     [5, 2, 1,  1, 0]])).ineqs_satisfied(points)
            True

            Points in 2d

            >>> points = numpy.array([[1, 1], [4, 2]])
            >>> ge_polyhedron(numpy.array([
            ...     [ 0, 1, 0],
            ...     [ 0, 1,-1],
            ...     [-1,-1, 1]])).ineqs_satisfied(points)
            boolean_ndarray([1, 0])

            Points in 3d

            >>> points = numpy.array([
            ...     [[1, 1, 1],
            ...      [4, 2, 1]],
            ...     [[0, 1, 0],
            ...      [1, 2, 1]]])
            >>> ge_polyhedron(numpy.array([
            ...     [ 0, 1, 0,-1],
            ...     [ 0, 1,-1, 0],
            ...     [-1,-1, 1,-1]])).ineqs_satisfied(points)
            boolean_ndarray([[1, 0],
                             [0, 0]])

        """
        if points.ndim > 2:
            return boolean_ndarray(
                list(
                    map(self.ineqs_satisfied, points)
                )
            )
        elif points.ndim == 2:
            return boolean_ndarray((numpy.dot(self.A, points.T) >= self.b[:,None]).all(axis=0))
        elif points.ndim == 1:
            return ge_polyhedron.ineqs_satisfied(self, numpy.array([points]))[0] == 1

    
    def column_bounds(self) -> "integer_ndarray":

        """
            Returns the initial bounds for each variable.

            Returns
            -------
                out : :class:`numpy.ndarray`
        """
        return integer_ndarray(
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
    def n_row_combinations(self) -> numpy.ndarray:

        """
            The number of combinations for variables with non-zero coefficients row wise excluding its constraint.

            Returns
            -------
                out : :class:`numpy.ndarray`

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

    def tighten_column_bounds(self) -> "integer_ndarray":
        
        """
            Returns maybe tighter column/variable bounds based on row constraints.

            Notes
            -----
            - Array returned has two rows - first is the lower bound and the second is the upper bound, on each column.
            - Lower bound may be larger than upper bound. This implies contradiction but no exception is raised here.

            Examples
            --------
                >>> ge_polyhedron([[0,-2,1,1,0],[1,1,0,0,0],[1,0,0,0,1],[-3,0,0,0,-1]]).tighten_column_bounds()
                integer_ndarray([[1, 0, 0, 1],
                                 [1, 1, 1, 1]])

                >>> ge_polyhedron([[3,1,1,1,0]]).tighten_column_bounds()
                integer_ndarray([[1, 1, 1, 0],
                                 [1, 1, 1, 1]])

                >>> ge_polyhedron([[0,-1,-1,0,0]]).tighten_column_bounds()
                integer_ndarray([[0, 0, 0, 0],
                                 [0, 0, 1, 1]])

                >>> ge_polyhedron([[2,-1,-1,0]], variables=[puan.variable(0, dtype="int"), puan.variable("a", bounds=(-2,1)), puan.variable("b"), puan.variable("c")]).tighten_column_bounds()
                integer_ndarray([[-2,  0,  0],
                                 [-2,  0,  1]])

            Returns
            -------
                out : :class:`integer_ndarray`
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

    def row_bounds(self) -> "integer_ndarray":

        """
            Returns the row equation bounds, including the bias.
            For instance, :math:`x+y+z\\ge1` has bounds of (-1,2).

            Examples
            --------
            >>> ge_polyhedron(numpy.array([[-2, -3,  0,  0,  0]])).row_bounds()
            integer_ndarray([[-1,  2]])

            Returns
            -------
                out : :class:`integer_ndarray`
        """
        clm_bounds = self.column_bounds()
        A_ = integer_ndarray([
            clm_bounds[0]*self.A,
            clm_bounds[1]*self.A,
        ])
        return integer_ndarray([
            A_.min(axis=0).sum(axis=1)-self.b, 
            A_.max(axis=0).sum(axis=1)-self.b
        ]).T

    def row_distribution(self, row_index: int) -> "integer_ndarray":

        """
            Returns a distribution of all combinations for each row and their values.
            Data type is a 2d numpy array with two columns: 
            Column index 0 is a range from lowest to highest value spot of row equation. 
            Column index 1 is a counter of how many combinations, generated from variable bounds, evaluated into that value spot.

        
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
                integer_ndarray([[0, 1],
                                 [1, 2],
                                 [2, 1]])

                >>> ge_polyhedron([[0,1,1,0],[0,2,2,0],[0,3,3,3]]).row_distribution(1)
                integer_ndarray([[0, 1],
                                 [1, 0],
                                 [2, 2],
                                 [3, 0],
                                 [4, 1]])

                >>> ge_polyhedron([[0,1,1,0],[0,2,2,0],[0,3,3,3]]).row_distribution(2)
                integer_ndarray([[0, 1],
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
                :meth:`row_stretch`      : Proportion of the number of value spots for each row equation.
                :meth:`row_stretch_int`  : Row bounds range from one number to another.

            Raises
            ------
                Exception
                    | Row index out of bounds.
                    | If variable mismatch between polyhedron variables and equation.

            Returns
            -------
                out : :class:`integer_ndarray`
        """
        if not (row_index >= 0 and row_index < self.shape[0]):
            raise Exception(f"row index {row_index} is out of bounds in polyhedron of shape {self.shape}")

        variable_bounds_full = integer_ndarray(
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

        return integer_ndarray([list(keys), list(values)]).T

    def row_stretch(self) -> "integer_ndarray":

        """
            This shows the proportion of the number of value spots for each
            row equation with respect to the number of combinations from active variable bounds. 
            It will return a vector of equal size as number of rows in polyhedron. Each value
            less than 1 means that there are more value spots than the number 
            of possible combinations for that row.

            See also
            --------
                :meth:`row_distribution` : Distribution of possible equation outcomes.
                :meth:`row_stretch_int`  : Row bounds range from one number to another.

            Examples
            --------
                >>> ge_polyhedron([[1,1,1,0,0],[2,2,2,0,0]]).row_stretch()
                integer_ndarray([1.33333333, 0.8       ])

            Returns
            -------
                out : :class:`integer_ndarray`
        """
        cm_bnds = self.column_bounds()
        rw_bnds = self.row_bounds()
        return numpy.prod((cm_bnds[1]-cm_bnds[0])*(self.A != 0)+1, axis=1) / (rw_bnds[:,1]-rw_bnds[:,0]+1)

    def row_stretch_int(self, row_index: int) -> int:

        """
            Row bounds range from one number to another. They normally
            have at least one combination for each number in this range.
            This function returns a number :math:`\leq0` for each row bound representing
            its "stretch". If the number is :math:`<0`, then there exists a gap in
            that row's bounds.

            As in example 2, there are no valid combinations summing up to the value spots 2 or 4. What this
            means is that the equation has unnecessary large coefficients which may have negative
            effects on other methods assuming a certain underlying equation structure.

            See also
            --------
                :meth:`row_distribution` : Distribution of possible equation outcomes.
                :meth:`row_stretch`      : Proportion of the number of value spots for each row equation.

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
            This operation uses :meth:`row_distribution` which will generate all combinations from variable bounds and may require heavy computation. 
            It is supposed to be used as an analysis tool. Use it with caution.

            Returns
            -------
                out : ``int``
        """
        rng, n = self.row_distribution(row_index).T
        return int((n != 0).sum() - (rng[-1]-rng[0]) -1)

class integer_ndarray(variable_ndarray):
    """
        A :class:`numpy.ndarray` sub class with only integers in it.

        Attributes
        ----------
            See : :class:`numpy.ndarray`

        Methods
        -------
        ndint_compress
        from_list
        get_neighbourhood
        ranking
        reduce2d
    """


    def reduce2d(self, method: typing.Literal["first", "last"]="first", axis: int=0) -> numpy.ndarray:
        """
            Reduces integer ndarray to only keeping one value of the given axis (default is 0) according to the provided method.

            Parameters
            ----------
            method : Literal["first", "last"]
                Which value to keep. Default "first".
            axis : ``int``
                Default 0.
            
            Raises
            ------
                ValueError
                    | If dimension is not 2
                    | If method is not 'first' or 'last'

            Examples
            --------
            >>> integer_ndarray([[1, 2, 3], [4, 5, 6]]).reduce2d()
            integer_ndarray([[1, 2, 3],
                             [0, 0, 0]])
            >>> integer_ndarray([[1, 2, 3], [4, 5, 6]]).reduce2d(method="last", axis=1)
            integer_ndarray([[0, 0, 3],
                             [0, 0, 6]])
        """
        if not self.ndim == 2:
            raise ValueError(f"ndarray must be 2d, is {self.ndim}d")
        self = numpy.swapaxes(self, 0, axis)
        col_idxs = numpy.arange(self.shape[1])
        self_reduced = integer_ndarray(numpy.zeros(self.shape))
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

    def ranking(self) -> "integer_ndarray":
        """
            Ranks an :class:`integer_ndarray`.

            Examples
            --------
            >>> integer_ndarray([[1, 0, 5], [2, 2, 6]]).ranking()
            integer_ndarray([[1, 0, 2],
                             [1, 1, 2]])
            
            Returns
            -------
                out : :class:`integer_ndarray`
        """
        if self.ndim > 1:
            return integer_ndarray(list(map(integer_ndarray.ranking, self)))
        else:
            # self.ndim == 1
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

    def ndint_compress(self, method: typing.Literal["first", "last", "min", "max", "prio", "rank", "shadow"]="min", axis: int=None) -> "integer_ndarray":

        """
            Takes an integer ndarray and compresses it into a vector under different conditions given by `method`.

            Parameters
            ----------
                method : {'first', 'last', 'min', 'max', 'prio', 'rank', 'shadow'}, optional
                    The method used to compress the `integer ndarray`. The following methods are available (default is 'min')

                    - 'min' Takes the minimum non-zero value
                    - 'max' Takes the maximum value
                    - 'last' Takes the last value
                    - 'prio' Gives the normalized prio of the input
                    - 'rank' Gives the rank of the input
                    - 'shadow' Treats the values as prioritizations as for 'prio' but the result value of a higher prioritization totally shadows lower priorities
                axis : {None, int}, optional
                    Axis along which to perform the compressing. If None, the data array is first flattened.

            Returns
            -------
                out : :class:`numpy.ndarray`
                    If input integer ndarray is input array is ``M-D`` the returned array is ``(M-1)-D``
            
            Raises
            ------
                ValueError
                    If method is not one of 'first', 'last', 'min', 'max', 'prio', 'rank' or 'shadow'

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

                >>> integer_ndarray([
                ...     [1, -2, 0, 0],
                ...     [0,  3, 4, 0],
                ...     [5,  0, 0, -6]]).ndint_compress(method='prio', axis=0)
                integer_ndarray([ 3,  1,  2, -4])

                >>> integer_ndarray([
                ...     [1, -2, 0, 0],
                ...     [0,  3, 4, 0],
                ...     [5,  0, 0, -6]]).ndint_compress(method='prio', axis=1)
                integer_ndarray([-1,  2, -3])

            Method 'rank'
                >>> integer_ndarray([1, 2, 1, 0, 4, 4, 6]).ndint_compress(method='rank')
                integer_ndarray([1, 2, 1, 0, 3, 3, 4])

                >>> integer_ndarray([
                ...     [1, -2, 0, 0],
                ...     [0,  3, 4, 0],
                ...     [5,  0, 0, 6]]).ndint_compress(method='rank', axis=0)
                integer_ndarray([3, 1, 2, 4])

                >>> integer_ndarray([
                ...     [1, -2, 0, 0],
                ...     [0,  3, 4, 0],
                ...     [5,  0, 0, 6]]).ndint_compress(method='rank', axis=1)
                integer_ndarray([0, 1, 2])

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
            self = numpy.swapaxes(self, 0, axis)
            if self.ndim > 2:
                return numpy.swapaxes(integer_ndarray(
                            list(
                                map(
                                    lambda x: integer_ndarray.ndint_compress(x, method=method, axis=0),
                                    self
                                )
                            )
                        ), 0, axis)
            elif self.ndim == 2:
                return numpy.flipud(self).ndint_compress(method="first", axis=0)
            else:
                return numpy.swapaxes(self, 0, axis-1)
        elif method == "first":
            self = numpy.swapaxes(self, 0, axis)
            if self.ndim > 2:
                return numpy.swapaxes(integer_ndarray(
                            list(
                                map(
                                    lambda x: integer_ndarray.ndint_compress(x, method=method, axis=0),
                                    self
                                )
                            )
                        ), 0, axis)
            elif self.ndim == 2:
                self = self[numpy.argmax(self!=0, axis=0),numpy.arange(self.shape[1])]
                return numpy.swapaxes(self, 0, axis-1)
            else:
                # self.ndim == 1
                return numpy.swapaxes(self, 0, axis-1)
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
                                    lambda x: integer_ndarray.ndint_compress(x, method=method, axis=0),
                                    self
                                )
                            )
                        ), 0, axis)
            elif self.ndim == 2:
                self_reduced = integer_ndarray(self).reduce2d(method="last", axis=0)
                # Convert negatives to positives
                self_reduced_abs = numpy.abs(self_reduced)
                # Remove zero rows
                self_reduced_abs = self_reduced_abs[~numpy.all(self_reduced_abs == 0, axis=1)]
                if self_reduced_abs.shape[0] == 0:
                    return integer_ndarray(numpy.zeros(self.shape[1], dtype=numpy.int64))
                self_reduced_abs = integer_ndarray(self_reduced_abs.ranking())
                self_reduced_abs = self_reduced_abs + ((self_reduced_abs.T>0) * numpy.concatenate(([0], (numpy.cumsum(self_reduced_abs.max(axis=1)))))[:-1]).T
                prio= self_reduced_abs.ndint_compress(method="first", axis=0)
                prio[self.ndint_compress(method="last", axis=0) < 0] = prio[self.ndint_compress(method="last", axis=0) < 0] * -1
                return prio

            else:
                # self.ndim == 1:
                return self.ranking()
        elif method == "rank":
            self = numpy.swapaxes(self, 0, axis)
            if self.ndim > 2:
                return numpy.swapaxes(integer_ndarray(
                            list(
                                map(
                                    lambda x: integer_ndarray.ndint_compress(x, method=method, axis=0),
                                    self
                                )
                            )
                        ), 0, axis)
            elif self.ndim == 2:
                return integer_ndarray(self.ndint_compress(method="prio", axis=0).ranking())
            else:
                # self.ndim == 1:
                return self.ranking()
        elif method == "shadow":
            self = numpy.swapaxes(self, 0, axis)
            if self.ndim > 2:
                return numpy.swapaxes(integer_ndarray(
                            list(
                                map(
                                    lambda x: integer_ndarray.ndint_compress(x, method=method, axis=0),
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
            else:
                # self.ndim == 1
                return integer_ndarray.ndint_compress(numpy.array([self], dtype=numpy.int64), method=method, axis=0)
        else:
            raise(ValueError("Method not recoginized, must be one of 'first', 'last', 'min', 'max', 'rank', 'shadow', got: {}".format(method)))

    def get_neighbourhood(self, method: typing.Literal["all", "addition", "subtraction"]="all", delta: typing.Union[int, numpy.ndarray]=1) -> "integer_ndarray":
        """
            Computes all neighbourhoods to an :class:`integer_ndarray`. A neighbourhood to the :class:`integer_ndarray` :math:`x` is defined as the
            :class:`integer_ndarray` :math:`y` which for one variable differs by *delta*, the values for the other variables are identical.

            Parameters
            ----------
                method : {'all', 'addition', 'subtraction'}, optional
                    The method used to compute the neighbourhoods to the integer ndarray. The following methods are available (default is 'all')

                    - 'all' computes all neighbourhoods, *addition* and *subtraction*.
                    - 'addition' computes all neighbourhoods with *delta* added
                    - 'subtraction' computes all neighbourhoods with *delta* subtracted

                delta : ``int`` or :class:`numpy.ndarray`, optional
                    The value added or subtracted for each variable to reach the neighbourhood, default is 1.
                    If delta is given as an array it is interpreted as different deltas for each variable.

            Returns
            -------
                out : :class:`integer_ndarray`
                    If input dimension is ``M-D`` the returned :class:`integer_ndarray` is ``M+1-D``

            Examples
            --------
                >>> x = integer_ndarray([0,0,0])
                >>> x.get_neighbourhood()
                integer_ndarray([[ 1,  0,  0],
                                 [ 0,  1,  0],
                                 [ 0,  0,  1],
                                 [-1,  0,  0],
                                 [ 0, -1,  0],
                                 [ 0,  0, -1]])

                >>> x = integer_ndarray([[0,0,0], [0,0,0]])
                >>> x.get_neighbourhood()
                integer_ndarray([[[ 1,  0,  0],
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
                integer_ndarray([[ 3,  1,  2],
                                 [ 0,  4,  2],
                                 [ 0,  1,  5],
                                 [-3,  1,  2],
                                 [ 0, -2,  2],
                                 [ 0,  1, -1]])
                >>> x.get_neighbourhood(delta=numpy.array([1, 2, 3]))
                integer_ndarray([[ 1,  1,  2],
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
        return integer_ndarray(numpy.concatenate((inc_neighbourhood, dec_neighbourhood), axis=self.ndim-1))

    @staticmethod
    def from_list(lst: typing.List[str], context: typing.List[str]) -> "integer_ndarray":
        """
            Turns a list (or list of list) of strings into an integer vector, where each value represents
            which order the string in lst was positioned.

            Parameters
            ----------
            lst     : List[str]
            context : List[str]

            Returns
            -------
                out : :class:`integer_ndarray`

            Examples
            --------
                >>> variables = ["a","c","b"]
                >>> context = ["a","b","c","d"]
                >>> integer_ndarray.from_list(variables, context)
                integer_ndarray([1, 3, 2, 0])

                >>> variables  = [["a", "c"], ["c", "b"], ["a", "b", "d"]]
                >>> context = ["a","b","c","d"]
                >>> integer_ndarray.from_list(variables, context)
                integer_ndarray([[1, 0, 2, 0],
                                 [0, 2, 1, 0],
                                 [1, 2, 0, 3]])

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
        A :class:`numpy.ndarray` sub class with only booleans in it.

        Attributes
        ----------
            See : :class:`numpy.ndarray`

        Methods
        -------
        from_list
        to_list
        get_neighbourhood
    """

    @staticmethod
    def from_list(lst: typing.List[str], context: typing.List[str]) -> "boolean_ndarray":
        """
            Turns a list of strings into a boolean (0/1) vector.

            Parameters
            ----------
            lst : List[str]
                list of variables ids

            context : List[str]
                list of context variables ids

            Returns
            -------
                out : :class:`boolean_ndarray`
                    booleans with same dimension as **context**

            Examples
            --------
                >>> variables = ["a","d","b"]
                >>> context = ["a","b","c","d"]
                >>> boolean_ndarray.from_list(variables, context)
                boolean_ndarray([1, 1, 0, 1])

                >>> variables  = [["a", "c"], ["c", "b"], ["a", "b", "d"]]
                >>> context = ["a","b","c","d"]
                >>> boolean_ndarray.from_list(variables, context)
                boolean_ndarray([[1, 0, 1, 0],
                                 [0, 1, 1, 0],
                                 [1, 1, 0, 1]])

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
                out : List[:class:`puan.variable`]
            
            Examples
            --------
                >>> boolean_ndarray([1, 1, 0, 1]).to_list()
                [variable(id=0, bounds=Bounds(lower=1, upper=1)), variable(id=1, bounds=Bounds(lower=0, upper=1)), variable(id=3, bounds=Bounds(lower=0, upper=1))]

                >>> boolean_ndarray([[1, 0, 1, 0],
                ...                  [0, 1, 1, 0],
                ...                  [1, 1, 0, 1]]).to_list()
                [[variable(id=0, bounds=Bounds(lower=1, upper=1)), variable(id=2, bounds=Bounds(lower=0, upper=1))], [variable(id=1, bounds=Bounds(lower=0, upper=1)), variable(id=2, bounds=Bounds(lower=0, upper=1))], [variable(id=0, bounds=Bounds(lower=1, upper=1)), variable(id=1, bounds=Bounds(lower=0, upper=1)), variable(id=3, bounds=Bounds(lower=0, upper=1))]]
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

    def get_neighbourhood(self, method: typing.Literal["on_off", "on", "off"]="on_off") -> "boolean_ndarray":
        """
            Computes all neighbourhoods to a :class:`boolean_ndarray`. A neighbourhood to the boolean ndarray :math:`x` is defined as the
            boolean ndarray :math:`y` which for one and only one variable in :math:`x` is the complement.

            Parameters
            ----------
                method : {'on_off', 'on', 'off'}, optional
                    The method used to compute the neighbourhoods to the boolean ndarray. The following methods are available (default is 'on_off')

                    - 'on_off' the most natural way to define neighbourhoods, if a variable is True in the input it is False in its neighbourhood and vice versa.
                    - 'on' do not include neighbourhoods with 'off switches', i.e. if a variable is True there is no neighbourhood to this variable.
                    - 'off' do not include neighbourhoods with 'on switches', i.e. if a variable is False there is no neighbourhood to this variable.
            
            Raises
            ------
                ValueError
                    If method is not one of 'on_off', 'on' or 'off' 

            Returns
            -------
                out : :class:`boolean_ndarray`
                    If input dimension is ``M-D`` the returned integer ndarray is ``M+1-D``

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
            raise(ValueError("Method not recognized, must be one of 'on_off', 'on' or 'off' for {}".format(method)))
        return _res[~(_res==self).all(axis=self.ndim-1)]

class InfeasibleError(Exception):
    """In context of solving on polyhedron and no feasible solution exists"""
    pass

class ge_polyhedron_config(ge_polyhedron):

    """
        A :class:`ge_polyhedron` sub class with specific configurator features.

        Methods
        -------
        select : select items to prioritize and receives a solution for each in prios list.
        to_b64 : packs data into a base64 string.
        from_b64 : unpacks base64 string `base64_str` into some data.
    """

    def __new__(cls, input_array: numpy.ndarray, default_prio_vector: numpy.ndarray = None, variables: typing.List[puan.variable] = [], index: typing.List[typing.Union[int, puan.variable]] = [], dtype=numpy.int64):
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
        solver: typing.Callable[[ge_polyhedron, typing.Iterable[numpy.ndarray]], typing.Iterable[typing.Tuple[typing.Optional[numpy.ndarray], typing.Optional[int], int]]] = None,
    ) -> itertools.starmap:

        """
            Select items to prioritize and receives a solution for each in prios list.

            Parameters
            ----------
                *prios : typing.List[typing.Dict[str, int]]
                    a list of dicts where each entry's value is a prio

                solver: typing.Callable[[ge_polyhedron, typing.Dict[str, int]], typing.List[(np.ndarray, int, int)]] = None
                    If None is provided puan's own (beta) solver is used. If you want to provide another solver
                    you have to send a function as solver parameter. That function has to take a :class:`ge_polyhedron` and
                    a 2d numpy array representing all objectives, as input. NOTE that the polyhedron **does not provide constraints for variable
                    bounds**. Variable bounds are found under each variable under `polyhedron.variables` and constraints for 
                    these has to manually be created and added to the polyhedron matrix. The function should return a list, one for each
                    objective, of tuples of (solution vector, objective value, status code). The solution vector is an integer ndarray vector
                    of size equal to width of ``polyhedron.A``. There are six different status codes from 1-6:

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
                >>> list(ph.select({"1": 1}))[0]
                ({1: 1, 2: 0, 3: 0, 4: 0}, -1, 5)

                >>> ph = ge_polyhedron_config([[1,1,1,1,0],[-1,-1,-1,-1,0]])
                >>> dummy_solver = lambda x, y: list(map(lambda v: (v, 0, 5), y))
                >>> list(ph.select({"1": 1}, solver=dummy_solver))[0]
                ({1: -1, 2: -1, 3: -1, 4: -1}, 0, 5)

            Raises
            ------
            InfeasibleError
                No solution could be found. Note that if solver raises another
                error, it will be shown within parantheses.

            Returns
            -------
                out : :class:`itertools.starmap`
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
                        pr.PolyhedronPy(
                            pr.MatrixPy(
                                self.A.flatten().tolist(),
                                *self.A.shape
                            ),
                            self.b.tolist(),
                            list(
                                map(
                                    lambda variable: pr.VariableFloatPy(
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

            return itertools.starmap(
                lambda solution, objective_value, status_code: (
                    dict(
                        zip(
                            map(
                                operator.attrgetter("id"),
                                variables
                            ),
                            solution
                        )
                    ) if solution is not None else {},
                    objective_value, status_code
                ),
                solutions,
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
                out : ``str``
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
            Unpacks base64 string ``base64_str``.

            Parameters
            ----------
                base64_str: ``str``

            Raises
            ------
                Exception
                    When error occurred during decompression.

            Returns
            -------
                out : :class:`ge_polyhedron_config`
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
    puan.ndarray.* on first level (e.g puan.ndarray.to_linalg())
"""
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
