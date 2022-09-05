#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Python.h>
#include <stdbool.h>
#define MIN(a,b) (((a)<=(b))?(a):(b))

static PyObject* Py_NewRef(PyObject* obj){
    Py_INCREF(obj);
    return obj;
}

static long acp_type(long b, long N, long sign){
    /* Categorize the logical relationship of an inequality given by b, N and sign as
    sign * x_1 + sign * x_2 + ... + sign * x_N >= b */
    if (sign > 0) // Require atleast b
        if (b > N) // Always false, not possible to satisfy the inequality
            return -1;
        else if (b==0) // Always true
            return 0;
        else if (b==1) // Atleast 1 <=> Any
            return 1;
        else if (N-b==0) // Atleast N <=> All
            return 4;
        else
            return 5; // Atleast n
    else // require atmost b
        if (b > 0)
            return -1; // Always false, not possible to satisfy the inequality
        else if (b <= -N)
            return 0; // Always true
        else if (b==0)
            return 3; // Atmost 0 <=> None
        else if (labs(b) == N-1)
            return 2; // Atmost all but one
        else
            return 6; // Atmost n

}

static long mcc_type(PyObject* constraint){ //constraint = ((indices),(constants),support, parent_index)
    PyObject* constants = PyTuple_GetItem(constraint, 2);
    long N = (long)PyTuple_Size(constants);
    long const1 = PyLong_AsLong(PyTuple_GetItem(constants, 0));
    if (const1 == 1 || const1 == -1){
        for (long i = 1; i < N; i++){
            if (const1 != PyLong_AsLong(PyTuple_GetItem(constants, i)))
            {
                return 7;
            }
        }
        return acp_type(PyLong_AsLong(PyTuple_GetItem(constraint, 3)), N, const1);
    }
    else {
        return 7;
    }
}

static PyObject* build_PyConstraintTuple(PyObject* var_type1, PyObject* indices1, long constant1, PyObject* var_type2, PyObject* indices2, long constant2, long b, PyObject* constr_id){
    PyObject* constraint_tuple = PyTuple_New(5);
    if (constraint_tuple == NULL)
        return NULL;
    long n1 = (long)PyTuple_Size(indices1);
    long n2 = (long)PyTuple_Size(indices2);
    PyObject* var_types = PyTuple_New(n1+n2);
    if (var_types == NULL){
        Py_DECREF(constraint_tuple);
        return NULL;
    }
    PyObject* indices = PyTuple_New(n1+n2);
    if (indices == NULL){
        Py_DECREF(constraint_tuple);
        Py_DECREF(var_types);
        return NULL;
    }
    PyObject* values = PyTuple_New(n1+n2);
    if (values == NULL){
        Py_DECREF(constraint_tuple);
        Py_DECREF(var_types);
        Py_DECREF(indices);
        return NULL;
    }
    for (long i = 0; i < n1; i++){
        PyTuple_SetItem(var_types, i, Py_NewRef(PyTuple_GetItem(var_type1, i)));
        PyTuple_SetItem(indices, i, Py_NewRef(PyTuple_GetItem(indices1, i)));
        PyTuple_SetItem(values, i, PyLong_FromLong(constant1));
    }
    for (long i=0; i < n2; i++){
        PyTuple_SetItem(var_types, n1+i, Py_NewRef(PyTuple_GetItem(var_type2, i)));
        PyTuple_SetItem(indices, n1+i, Py_NewRef(PyTuple_GetItem(indices2, i)));
        PyTuple_SetItem(values, n1+i, PyLong_FromLong(constant2));
    }
    PyTuple_SetItem(constraint_tuple, 0, var_types);
    PyTuple_SetItem(constraint_tuple, 1, indices);
    PyTuple_SetItem(constraint_tuple, 2, values);
    PyTuple_SetItem(constraint_tuple, 3, PyLong_FromLong(b));
    PyTuple_SetItem(constraint_tuple, 4, Py_NewRef(constr_id));
    return constraint_tuple;
};

static PyObject* build_PyCompoundConstraints(long b, long sign, PyObject* constraint_list, PyObject* constr_id){
    PyObject* result = PyTuple_New(4);
    if (result == NULL)
        return NULL;
    PyTuple_SetItem(result, 0, PyLong_FromLong(b));
    PyTuple_SetItem(result, 1, PyLong_FromLong(sign));
    PyTuple_SetItem(result, 2, Py_NewRef(constraint_list));
    PyTuple_SetItem(result, 3, Py_NewRef(constr_id));
    return result;
}

long lower_bound(PyObject* constraint){
    PyObject* constants = PyTuple_GetItem(constraint, 2);
    long b = PyLong_AsLong(PyTuple_GetItem(constraint, 3));
    long N = (long) PyTuple_Size(constants);
    long res = -b;
    for (int i = 0; i < N; i++){
        if (PyLong_AsLong(PyTuple_GetItem(constants, i)) < 0)
            res--;
    }
    return res;
}

static PyObject* generate_result(PyObject* constraint, long b, long sign, PyObject* constr_id){
    /* Note! This function does NOT increment the reference count for the PyObject constraint before adding it
    to the constraint_list */
    PyObject* constraint_list = PyList_New(1);
    if (constraint_list == NULL)
        return NULL;
    PyList_SetItem(constraint_list, 0, constraint);
    PyObject *return_this = build_PyCompoundConstraints(b, sign, constraint_list, constr_id);
    Py_DECREF(constraint_list);
    return return_this;
}

static PyObject* compress_two_disjunctions(PyObject* constraint1, PyObject* constraint2, PyObject* constr_id){
    /* Takes two constraints and their constraint identifier and returns a new constraint */
    long ctype1 = mcc_type(constraint1);
    long ctype2 = mcc_type(constraint2);
    long n1 = (long)PyTuple_Size(PyTuple_GetItem(constraint1, 2));
    long n2 = (long)PyTuple_Size(PyTuple_GetItem(constraint2, 2));

    if (ctype1 == -1 && ctype2 == -1){
        PyErr_SetString(PyExc_ValueError, "ERROR: Found contradicition");
        Py_RETURN_NONE;
    } else if (ctype1 == -1){
        return generate_result(Py_NewRef(constraint2), 1L, 1L, constr_id);
    } else if (ctype2 == -1){
        return generate_result(Py_NewRef(constraint1), 1L, 1L, constr_id);
    } else if (ctype1 == 0 || ctype2 == 0){
        /* Both constraints are redundant */
        return generate_result(Py_NewRef(Py_None), 0L, 1L, constr_id);
    } else if (ctype1 > ctype2){
        return compress_two_disjunctions(constraint2, constraint1, constr_id);
    } else if (ctype1 == 1){
        if (ctype2 == 1){
            return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), 1, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), 1, 1, constr_id), 1L, 1L, constr_id);
        } else if (ctype2 == 2 ){
            return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), n2, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), -1, 0, constr_id), 1L, 1L, constr_id);
        } else if (ctype2 == 3 || ctype2 == 6){
            return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1),  n2, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), -1, PyLong_AsLong(PyTuple_GetItem(constraint2, 3)), constr_id), 1L, 1L, constr_id);
        } else if (ctype2 == 4 || ctype2 == 5){
            return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), n2, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), 1, PyLong_AsLong(PyTuple_GetItem(constraint2, 3)), constr_id), 1L, 1L, constr_id);
        } else {
            PyObject* constraint_list = PyList_New(2);
            if (constraint_list == NULL)
                return NULL;
            PyList_SetItem(constraint_list, 0, Py_NewRef(constraint1));
            PyList_SetItem(constraint_list, 1, Py_NewRef(constraint2));
            PyObject *return_this = build_PyCompoundConstraints(1, 1, constraint_list, constr_id);
            Py_DECREF(constraint_list);
            return return_this;
        }
    } else if (ctype1 == 2){
        if (ctype2 == 2){
            return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), -n2, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), -1, -n2*(labs(PyLong_AsLong(PyTuple_GetItem(constraint1, 3)))+1)-labs(PyLong_AsLong(PyTuple_GetItem(constraint2, 3))), constr_id), 1L, 1L, constr_id);
        } else if (ctype2 == 3){
            return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), -1, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), -n1, -n1*n2, constr_id), 1L, 1L, constr_id);
        } else if (ctype2 == 4 || ctype2 == 5){
            return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), -n2, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), 1, -n2*(labs(PyLong_AsLong(PyTuple_GetItem(constraint1, 3)))+1)+PyLong_AsLong(PyTuple_GetItem(constraint2, 3)), constr_id), 1L, 1L, constr_id);
        } else if (ctype2 == 6){
            return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), -n2, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), -1, -n2*(labs(PyLong_AsLong(PyTuple_GetItem(constraint1, 3)))+1)+PyLong_AsLong(PyTuple_GetItem(constraint2, 3)), constr_id), 1L, 1L, constr_id);
        } else {
            PyObject* constraint_list = PyList_New(2);
            if (constraint_list ==  NULL)
                return NULL;
            PyList_SetItem(constraint_list, 0, Py_NewRef(constraint1));
            PyList_SetItem(constraint_list, 1, Py_NewRef(constraint2));
            PyObject* return_this = build_PyCompoundConstraints(1, 1, constraint_list, constr_id);
            Py_DECREF(constraint_list);
            return return_this;
        }
    } else if (ctype1 == 3 || ctype1 == 4){
        if (n1 == 1){
            if (ctype2 == 3){
                return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), -n2, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), -1, -n2, constr_id), 1L, 1L, constr_id);
            } else if (ctype2 == 4){
                return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), -n2, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), 1, 0, constr_id), 1L, 1L, constr_id);
            } else if (ctype2 == 5){
                return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), -n2, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), 1, -n2*PyLong_AsLong(PyTuple_GetItem(constraint2, 3)), constr_id), 1L, 1L, constr_id);
            } else if (ctype2 == 6){
                return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), -n2, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), -1, -n2+PyLong_AsLong(PyTuple_GetItem(constraint2, 3)), constr_id), 1L, 1L, constr_id);
            } else {
                PyObject* constraint_list = PyList_New(2);
                if (constraint_list ==  NULL)
                    return NULL;
                PyList_SetItem(constraint_list, 0, Py_NewRef(constraint1));
                PyList_SetItem(constraint_list, 1, Py_NewRef(constraint2));
                PyObject* return_this = build_PyCompoundConstraints(1, 1, constraint_list, constr_id);
                Py_DECREF(constraint_list);
                return return_this;
            }
        } else if (n2 == 1 && (ctype2 == 3 || ctype2 == 4)){
            if (ctype1 == 3){
                return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), -1, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), -n1, -n1, constr_id), 1L, 1L, constr_id);
            } else {
                return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), 1, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), -n1, 0, constr_id), 1L, 1L, constr_id);
            }
        } else{
            PyObject* constraint_list = PyList_New(2);
            if (constraint_list ==  NULL)
                return NULL;
            PyList_SetItem(constraint_list, 0, Py_NewRef(constraint1));
            PyList_SetItem(constraint_list, 1, Py_NewRef(constraint2));
            PyObject *return_this = build_PyCompoundConstraints(1, 1, constraint_list, constr_id);
            Py_DECREF(constraint_list);
            return return_this;
        }
    } else {
        PyObject* constraint_list = PyList_New(2);
        if (constraint_list ==  NULL)
            return NULL;
        PyList_SetItem(constraint_list, 0, Py_NewRef(constraint1));
        PyList_SetItem(constraint_list, 1, Py_NewRef(constraint2));
        PyObject* return_this = build_PyCompoundConstraints(1, 1, constraint_list, constr_id);
        Py_DECREF(constraint_list);
        return return_this;
    }
}

static PyObject* compress_disjunctions(PyObject* constraints, PyObject* constr_id){
    long N = (long) PyList_Size(constraints);
    PyObject * tmp;
    PyObject* res[N];
    long j = 0;
    PyObject* constraint1 = PyList_GetItem(constraints, 0);
    bool incr_refcnt = false;
    for (long i=1; i < N; i++){
        tmp = compress_two_disjunctions(constraint1, PyList_GetItem(constraints, i), constr_id);
        if (tmp == NULL)
            return NULL;
        if (PyList_Size(PyTuple_GetItem(tmp, 2)) < 2){
            constraint1 = Py_NewRef(PyList_GetItem(PyTuple_GetItem(tmp, 2), 0));
            incr_refcnt = false;
        } else{
            res[j] = Py_NewRef(constraint1);
            constraint1 = PyList_GetItem(constraints, i);
            j++;
            incr_refcnt = true;
        }
        Py_DECREF(tmp);
        tmp = NULL;
    }

    if (incr_refcnt)
        Py_INCREF(constraint1);
    res[j] = constraint1;
    PyObject* result = PyList_New(j+1);
    if (result == NULL){
        for(int i=0; i <= j; i++){
            Py_DECREF(res[i]);
        }
        return NULL;
    }
    for(int i=0; i <= j; i++){
        PyList_SetItem(result, i, res[i]);
    }
    PyObject* return_this = build_PyCompoundConstraints(1, 1, result, constr_id);
    Py_DECREF(result);
    return return_this;
}


static PyObject* create_new_variable(long b, long sign, PyObject* constraints, PyObject* constr_id){
    long N = (long) PyList_Size(constraints);
    long N_i;
    PyObject* constraint_list = PyList_New(N+1);
    if (constraint_list == NULL)
        return NULL;
    PyObject* constraint_i;
    PyObject* main_constraint_var_types = PyTuple_New(N);
    if (main_constraint_var_types == NULL){
        Py_DECREF(constraint_list);
        return NULL;
    }
    PyObject* main_constraint_indices = PyTuple_New(N);
    if (main_constraint_indices == NULL){
        Py_DECREF(constraint_list);
        Py_DECREF(main_constraint_var_types);
        return NULL;
    }
    PyObject* main_constraint_values = PyTuple_New(N);
    if (main_constraint_values == NULL){
        Py_DECREF(constraint_list);
        Py_DECREF(main_constraint_var_types);
        Py_DECREF(main_constraint_indices);
        return NULL;
    }
    long m;
    PyObject* var_type_support_constraint;
    PyObject* ind_support_constraint;
    PyObject* val_support_constraint;
    for (long i=0; i < N; i++){
        constraint_i = PyList_GetItem(constraints, i);
        N_i = (long) PyTuple_Size(PyTuple_GetItem(constraint_i, 0));
        PyTuple_SetItem(main_constraint_var_types, i, PyLong_FromLong(0));
        PyTuple_SetItem(main_constraint_indices, i, Py_NewRef(PyTuple_GetItem(constraint_i, 4)));
        PyTuple_SetItem(main_constraint_values, i, PyLong_FromLong(sign));
        m = lower_bound(constraint_i);
        var_type_support_constraint = PyTuple_New(1+N_i);
        if (var_type_support_constraint == NULL){
            Py_DECREF(constraint_list);
            Py_DECREF(main_constraint_var_types);
            Py_DECREF(main_constraint_indices);
            Py_DECREF(main_constraint_values);
            return NULL;
        }
        ind_support_constraint = PyTuple_New(1+N_i);
        if (ind_support_constraint == NULL){
            Py_DECREF(constraint_list);
            Py_DECREF(main_constraint_var_types);
            Py_DECREF(main_constraint_indices);
            Py_DECREF(main_constraint_values);
            Py_DECREF(var_type_support_constraint);
            return NULL;
        }
        val_support_constraint = PyTuple_New(1+N_i);
        if (val_support_constraint == NULL){
            Py_DECREF(constraint_list);
            Py_DECREF(main_constraint_var_types);
            Py_DECREF(main_constraint_indices);
            Py_DECREF(main_constraint_values);
            Py_DECREF(var_type_support_constraint);
            Py_DECREF(ind_support_constraint);
            return NULL;
        }
        for (int j=0; j < N_i; j++){
            PyTuple_SetItem(var_type_support_constraint, j, Py_NewRef(PyTuple_GetItem(PyTuple_GetItem(constraint_i, 0), j)));
            PyTuple_SetItem(ind_support_constraint, j, Py_NewRef(PyTuple_GetItem(PyTuple_GetItem(constraint_i, 1), j)));
            PyTuple_SetItem(val_support_constraint, j, Py_NewRef(PyTuple_GetItem(PyTuple_GetItem(constraint_i, 2), j)));
        }
        PyTuple_SetItem(var_type_support_constraint, N_i, PyLong_FromLong(0));
        PyTuple_SetItem(ind_support_constraint, N_i, Py_NewRef(PyTuple_GetItem(constraint_i, 4)));
        PyTuple_SetItem(val_support_constraint, N_i, PyLong_FromLong(m));
        PyObject* support_constraint_i = PyTuple_New(5);
        if (support_constraint_i == NULL){
            Py_DECREF(constraint_list);
            Py_DECREF(main_constraint_var_types);
            Py_DECREF(main_constraint_indices);
            Py_DECREF(main_constraint_values);
            Py_DECREF(var_type_support_constraint);
            Py_DECREF(ind_support_constraint);
            Py_DECREF(val_support_constraint);
            return NULL;
        }
        PyTuple_SetItem(support_constraint_i, 0, var_type_support_constraint);
        PyTuple_SetItem(support_constraint_i, 1, ind_support_constraint);
        PyTuple_SetItem(support_constraint_i, 2, val_support_constraint);
        PyTuple_SetItem(support_constraint_i, 3, PyLong_FromLong(m + PyLong_AsLong(PyTuple_GetItem(constraint_i, 3))));
        PyTuple_SetItem(support_constraint_i, 4, Py_NewRef(PyTuple_GetItem(constraint_i, 4)));
        PyList_SetItem(constraint_list, i+1, support_constraint_i);
    }
    PyObject* main_constraint = PyTuple_New(5);
    if (main_constraint == NULL){
        Py_DECREF(constraint_list);
        Py_DECREF(main_constraint_var_types);
        Py_DECREF(main_constraint_indices);
        Py_DECREF(main_constraint_values);
        Py_DECREF(var_type_support_constraint);
        Py_DECREF(ind_support_constraint);
        Py_DECREF(val_support_constraint);
        return NULL;
    }
    PyTuple_SetItem(main_constraint, 0, main_constraint_var_types);
    PyTuple_SetItem(main_constraint, 1, main_constraint_indices);
    PyTuple_SetItem(main_constraint, 2, main_constraint_values);
    PyTuple_SetItem(main_constraint, 3, PyLong_FromLong(b));
    PyTuple_SetItem(main_constraint, 4, Py_NewRef(constr_id));
    PyList_SetItem(constraint_list, 0, main_constraint);
    PyObject *return_this = build_PyCompoundConstraints(N+1, 1, constraint_list, constr_id);
    Py_DECREF(constraint_list);
    return return_this;
}

static void create_constraints(PyObject* constraint_list, PyObject* constraint1, long constant1, PyObject* constraint2, long constant2, long b, PyObject* constr_id){
    int N = (int)PyList_Size(constraint_list);
    PyObject *_var_type;
    PyObject *_index;
    for (int i=0; i < N; i++){
        _var_type = PyTuple_New(1);
        if (_var_type == NULL)
            return;
        _index = PyTuple_New(1);
        if (_index == NULL){
            Py_DECREF(_var_type);
            return;
        }
        PyTuple_SetItem(_var_type, 0, Py_NewRef(PyTuple_GetItem(PyTuple_GetItem(constraint1, 0), i)));
        PyTuple_SetItem(_index, 0, Py_NewRef(PyTuple_GetItem(PyTuple_GetItem(constraint1, 1), i)));
        PyList_SetItem(constraint_list, i, build_PyConstraintTuple(_var_type, _index, constant1, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), constant2, b, constr_id));
        Py_DECREF(_var_type);
        Py_DECREF(_index);
    }
}

static PyObject * transform_two_disjunctions(PyObject* constraint1, PyObject* constraint2, PyObject* constr_id, bool allow_mc){
    int ctype1 = mcc_type(constraint1);
    int ctype2 = mcc_type(constraint2);
    long n1 = (long)PyTuple_Size(PyTuple_GetItem(constraint1, 2));
    long n2 = (long)PyTuple_Size(PyTuple_GetItem(constraint2, 2));

    if (ctype1 > ctype2)
        return transform_two_disjunctions(constraint2, constraint1, constr_id, allow_mc);
    else if (ctype1 <= 2 || ctype2 <= 2)
        return compress_two_disjunctions(constraint1, constraint2, constr_id);
    else if (ctype1 == 3 || ctype1 == 4){
        if (n1 == 1)
            return compress_two_disjunctions(constraint1, constraint2, constr_id);
        else if (n2 == 1 && (ctype2 == 3 || ctype2 == 4))
            return compress_two_disjunctions(constraint1, constraint2, constr_id);
        else if (allow_mc && ctype2 <= 4){
            if (n2 < n1){
                PyObject* constraint_list = PyList_New(n2);
                if (constraint_list ==  NULL)
                    return NULL;
                if (ctype2 == 3){
                    if(ctype1 == 3)
                        create_constraints(constraint_list, constraint2, -n1, constraint1, -1L, -n1, constr_id);
                    else //ctype1 == 4
                        create_constraints(constraint_list, constraint2, -n1, constraint1, 1L, 0, constr_id);
                } else { // ctype2 == 4
                    if (ctype1 == 3)
                        create_constraints(constraint_list, constraint2, n1, constraint1, -1L, 0, constr_id);
                    else // ctype1 == 4
                        create_constraints(constraint_list, constraint2, n1, constraint1, 1L, n1, constr_id);
                }
                PyObject* return_this = build_PyCompoundConstraints(n2, 1, constraint_list, constr_id);
                Py_DECREF(constraint_list);
                return return_this;
            } else {
                PyObject* constraint_list = PyList_New(n1);
                if (constraint_list ==  NULL)
                    return NULL;
                if (ctype1 == 3){
                    if (ctype2 == 3)
                        create_constraints(constraint_list, constraint1, -n2, constraint2, -1L, -n2, constr_id);
                    else // ctype2 == 4
                        create_constraints(constraint_list, constraint1, -n2, constraint2, 1L, 0L, constr_id);
                } else // ctype1 == 4 and ctype2 == 4
                    create_constraints(constraint_list, constraint1, n2, constraint2, 1L, n2, constr_id);
                PyObject* return_this = build_PyCompoundConstraints(n1, 1, constraint_list, constr_id);
                Py_DECREF(constraint_list);
                return return_this;
            }
        } else if (allow_mc){
            PyObject* constraint_list = PyList_New(n1);
            if (constraint_list ==  NULL)
                return NULL;
            if (ctype1 == 3)
                if (ctype2 == 5)
                    create_constraints(constraint_list, constraint1, -n2, constraint2, 1L, -n2 + PyLong_AsLong(PyTuple_GetItem(constraint2, 3)), constr_id);
                else
                    create_constraints(constraint_list, constraint1, -n2, constraint2, -1L, -n2 + PyLong_AsLong(PyTuple_GetItem(constraint2, 3)), constr_id);
            else //ctype1 == 4
                if (ctype2 == 5)
                    create_constraints(constraint_list, constraint1, n2, constraint2, 1L, PyLong_AsLong(PyTuple_GetItem(constraint2, 3)), constr_id);
                else
                    create_constraints(constraint_list, constraint1, n2, constraint2, -1L, PyLong_AsLong(PyTuple_GetItem(constraint2, 3)), constr_id);
            PyObject* return_this = build_PyCompoundConstraints(n1, 1, constraint_list, constr_id);
            Py_DECREF(constraint_list);
            return return_this;
        } else {
            PyObject* constraint_list = PyList_New(2);
            if (constraint_list ==  NULL)
                return NULL;
            PyList_SetItem(constraint_list, 0, Py_NewRef(constraint1));
            PyList_SetItem(constraint_list, 1, Py_NewRef(constraint2));
            PyObject* return_this = create_new_variable(1, 1, constraint_list, constr_id);
            Py_DECREF(constraint_list);
            return return_this;
        }
    } else {
        PyObject* constraint_list = PyList_New(2);
        if (constraint_list ==  NULL)
            return NULL;
        PyList_SetItem(constraint_list, 0, Py_NewRef(constraint1));
        PyList_SetItem(constraint_list, 1, Py_NewRef(constraint2));
        PyObject* return_this = create_new_variable(1, 1, constraint_list, constr_id);
        Py_DECREF(constraint_list);
        return return_this;
    }

}

static PyObject* transform_disjunctions(PyObject* constraints, PyObject* constr_id, bool allow_mc)
{
    PyObject* compressed_compound_constraints = compress_disjunctions(constraints, constr_id);
    PyObject* compressed_constraints = PyTuple_GetItem(compressed_compound_constraints, 2);

    long N = (long) PyList_Size(compressed_constraints);
    if (N < 2){
        return compressed_compound_constraints;
    } else if (allow_mc && N < 3) {
        PyObject *return_this = transform_two_disjunctions(PyList_GetItem(compressed_constraints, 0), PyList_GetItem(compressed_constraints, 1), constr_id, allow_mc);
        Py_DECREF(compressed_compound_constraints);
        return return_this;
    } else {
        PyObject* return_this = create_new_variable(PyLong_AsLong(PyTuple_GetItem(compressed_compound_constraints, 0)), PyLong_AsLong(PyTuple_GetItem(compressed_compound_constraints, 1)), compressed_constraints, constr_id);
        Py_DECREF(compressed_compound_constraints);
        return return_this;
    }
}

static PyObject* compress_two_conjunctions(PyObject* constraint1, PyObject* constraint2, PyObject* constr_id){
    /* Returns a Python Compound Constraint */
    PyObject* constraint1_const = PyTuple_GetItem(constraint1, 2);
    PyObject* constraint2_const = PyTuple_GetItem(constraint2, 2);
    long ctype1 = mcc_type(constraint1);
    long ctype2 = mcc_type(constraint2);
    long n1 = (long)PyTuple_Size(constraint1_const);
    long n2 = (long)PyTuple_Size(constraint2_const);

    if (ctype1 == -1 || ctype2 == -1){
        PyErr_SetString(PyExc_ValueError, "ERROR: Found contradicition");
        Py_RETURN_NONE;
    } else if (ctype1 == 0){
        return generate_result(Py_NewRef(constraint2), 1L, 1L, constr_id);
    } else if (ctype2 == 0){
        return generate_result(Py_NewRef(constraint1), 1L, 1L, constr_id);
    } else if (ctype1 > ctype2){
        return compress_two_conjunctions(constraint2, constraint1, constr_id);
    } else if (ctype1 == 1){
        if (n1 == 1){ // treat it as ctype 4
            if (ctype2 == 1){
                return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), n2, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), 1, n1*n2 + PyLong_AsLong(PyTuple_GetItem(constraint2, 3)), constr_id), 1L, 1L, constr_id);
            } else if (ctype2 == 2){
                return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), n2, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), -1, n1*n2 + PyLong_AsLong(PyTuple_GetItem(constraint2, 3)), constr_id), 1L, 1L, constr_id);
            } else if (ctype2 == 3){
                return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), 1, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), -1, n1, constr_id), 1L, 1L, constr_id);
            } else if (ctype2==4){
                return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), 1, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), 1, n1+n2, constr_id), 1L, 1L, constr_id);
            } else if (ctype2==5){
                return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), n2, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), 1, n1*n2 + PyLong_AsLong(PyTuple_GetItem(constraint2, 3)), constr_id), 1L, 1L, constr_id);
            } else if (ctype2==6){
                return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), n2, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), -1, n1*n2 + PyLong_AsLong(PyTuple_GetItem(constraint2, 3)), constr_id), 1L, 1L, constr_id);
            } else {
                PyObject* constraint_list = PyList_New(2);
                if (constraint_list ==  NULL)
                    return NULL;
                PyList_SetItem(constraint_list, 0, Py_NewRef(constraint1));
                PyList_SetItem(constraint_list, 1, Py_NewRef(constraint2));
                PyObject* return_this = build_PyCompoundConstraints(1, 1, constraint_list, constr_id);
                Py_DECREF(constraint_list);
                return return_this;
            }
        } else if (ctype2 == 3){
            return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), 1, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), n1, PyLong_AsLong(PyTuple_GetItem(constraint1, 3)), constr_id), 1L, 1L, constr_id);
        } else if (ctype2 == 4 ){
            return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), 1, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), n1, n1*n2 + PyLong_AsLong(PyTuple_GetItem(constraint1, 3)), constr_id), 1L, 1L, constr_id);
        } else {
            PyObject* constraint_list = PyList_New(2);
            if (constraint_list ==  NULL)
                return NULL;
            PyList_SetItem(constraint_list, 0, Py_NewRef(constraint1));
            PyList_SetItem(constraint_list, 1, Py_NewRef(constraint2));
            PyObject* return_this = build_PyCompoundConstraints(1, 1, constraint_list, constr_id);
            Py_DECREF(constraint_list);
            return return_this;
        }
    } else if (ctype1 == 2){
        if (ctype2 == 3){
            return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), -1, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), -n1, PyLong_AsLong(PyTuple_GetItem(constraint1, 3)), constr_id), 1L, 1L, constr_id);
        } else if (ctype2 == 4){
            return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), -1, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), n1, n1*n2 + PyLong_AsLong(PyTuple_GetItem(constraint1, 3)), constr_id), 1L, 1L, constr_id);
        } else {
            PyObject* constraint_list = PyList_New(2);
            if (constraint_list ==  NULL)
                return NULL;
            PyList_SetItem(constraint_list, 0, Py_NewRef(constraint1));
            PyList_SetItem(constraint_list, 1, Py_NewRef(constraint2));
            PyObject* return_this = build_PyCompoundConstraints(1, 1, constraint_list, constr_id);
            Py_DECREF(constraint_list);
            return return_this;
        }
    } else if (ctype1 == 3){
        if (ctype2 == 3){
            return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), -1, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), -1, 0, constr_id), 1L, 1L, constr_id);
        } else if (ctype2 == 4){
            return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), -1, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), 1, n2, constr_id), 1L, 1L, constr_id);
        } else if (ctype2 == 5){
            return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), -n2, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), 1, PyLong_AsLong(PyTuple_GetItem(constraint2, 3)), constr_id), 1L, 1L, constr_id);
        } else if (ctype2 == 6){
            return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), -n2, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), -1, PyLong_AsLong(PyTuple_GetItem(constraint2, 3)), constr_id), 1L, 1L, constr_id);
        } else {
            PyObject* constraint_list = PyList_New(2);
            if (constraint_list ==  NULL)
                return NULL;
            PyList_SetItem(constraint_list, 0, Py_NewRef(constraint1));
            PyList_SetItem(constraint_list, 1, Py_NewRef(constraint2));
            PyObject* return_this = build_PyCompoundConstraints(1, 1, constraint_list, constr_id);
            Py_DECREF(constraint_list);
            return return_this;
        }
    } else if (ctype1==4){
        if (ctype2==4){
            return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), 1, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), 1, n1+n2, constr_id), 1L, 1L, constr_id);
        } else if (ctype2==5){
            return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), n2, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), 1, n1*n2 + PyLong_AsLong(PyTuple_GetItem(constraint2, 3)), constr_id), 1L, 1L, constr_id);
        } else if (ctype2==6){
            return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), n2, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), -1, n1*n2 + PyLong_AsLong(PyTuple_GetItem(constraint2, 3)), constr_id), 1L, 1L, constr_id);
        } else {
            PyObject* constraint_list = PyList_New(2);
            if (constraint_list ==  NULL)
                return NULL;
            PyList_SetItem(constraint_list, 0, Py_NewRef(constraint1));
            PyList_SetItem(constraint_list, 1, Py_NewRef(constraint2));
            PyObject* return_this = build_PyCompoundConstraints(1, 1, constraint_list, constr_id);
            Py_DECREF(constraint_list);
            return return_this;
        }
    } else if (ctype1==5){
        if (ctype2==6){
            return generate_result(build_PyConstraintTuple(PyTuple_GetItem(constraint1, 0), PyTuple_GetItem(constraint1, 1), n2, PyTuple_GetItem(constraint2, 0), PyTuple_GetItem(constraint2, 1), -1, PyLong_AsLong(PyTuple_GetItem(constraint1, 3))*n2 + PyLong_AsLong(PyTuple_GetItem(constraint2, 3)), constr_id), 1L, 1L, constr_id);
        } else {
            PyObject* constraint_list = PyList_New(2);
            if (constraint_list ==  NULL)
                return NULL;
            PyList_SetItem(constraint_list, 0, Py_NewRef(constraint1));
            PyList_SetItem(constraint_list, 1, Py_NewRef(constraint2));
            PyObject* return_this = build_PyCompoundConstraints(1, 1, constraint_list, constr_id);
            Py_DECREF(constraint_list);
            return return_this;
        }
    } else {
        PyObject* constraint_list = PyList_New(2);
        if (constraint_list ==  NULL)
            return NULL;
        PyList_SetItem(constraint_list, 0, Py_NewRef(constraint1));
        PyList_SetItem(constraint_list, 1, Py_NewRef(constraint2));
        PyObject* return_this = build_PyCompoundConstraints(1, 1, constraint_list, constr_id);
        Py_DECREF(constraint_list);
        return return_this;
    }
}

static PyObject* compress_conjunctions(PyObject* constraints, PyObject* constr_id){
    long N = (long) PyList_Size(constraints);
    PyObject * tmp;
    PyObject* res[N];
    long j = 0;
    PyObject* constraint1 = PyList_GetItem(constraints, 0);
    bool incr_refcnt = false;
    for (long i=1; i < N; i++){
        tmp = compress_two_conjunctions(constraint1, PyList_GetItem(constraints, i), constr_id);
        if (PyList_Size(PyTuple_GetItem(tmp, 2)) < 2){
            constraint1 = Py_NewRef(PyList_GetItem(PyTuple_GetItem(tmp, 2), 0));
            incr_refcnt = false;
        } else{
            res[j] = Py_NewRef(constraint1);
            constraint1 = PyList_GetItem(constraints, i);
            j++;
            incr_refcnt = true;
        }
        Py_DECREF(tmp);
        tmp = NULL;
    }
    if (incr_refcnt)
        Py_INCREF(constraint1);
    res[j] = constraint1;
    PyObject* result = PyList_New(j+1);
    if (result ==  NULL){
        for(int i=0; i <= j; i++){
            Py_DECREF(res[i]);
        }
        return NULL;
    }
    for(int i=0; i <= j; i++){
        PyList_SetItem(result, i, res[i]);
    }
    PyObject* return_this = build_PyCompoundConstraints(j+1, 1, result, constr_id);
    Py_DECREF(result);
    return return_this;
}

static PyObject * transform_two_conjunctions(PyObject* constraint1, PyObject* constraint2, PyObject* constr_id, bool allow_mc){
    int ctype1 = mcc_type(constraint1);
    int ctype2 = mcc_type(constraint2);
    int n1 = (int)PyTuple_Size(PyTuple_GetItem(constraint1, 2));
    int n2 = (int)PyTuple_Size(PyTuple_GetItem(constraint2, 2));

    if (ctype1 > ctype2)
        return transform_two_conjunctions(constraint2, constraint1, constr_id, allow_mc);
    else if (ctype1 == 3 || ctype1 == 4 || ctype2 == 3 || ctype2 == 4 || (ctype1 == 5 && ctype2 == 6))
        return compress_two_conjunctions(constraint1, constraint2, constr_id);
    else if (ctype1 == 1){
        if (allow_mc){
            PyObject* constraint_list;
            if (ctype2 == 1){
                if (n2 < n1){
                    constraint_list = PyList_New(n2);
                    if (constraint_list ==  NULL)
                        return NULL;
                    create_constraints(constraint_list, constraint2, n1, constraint1, 1L, n1+1, constr_id);
                } else {
                    constraint_list = PyList_New(n1);
                    if (constraint_list ==  NULL)
                        return NULL;
                    create_constraints(constraint_list, constraint1, n2, constraint2, 1L, n2+1, constr_id);
                }
            } else if (ctype2 == 5){
                constraint_list = PyList_New(n1);
                if (constraint_list ==  NULL)
                    return NULL;
                create_constraints(constraint_list, constraint1, n2, constraint2, 1L, n2+PyLong_AsLong(PyTuple_GetItem(constraint2, 3)), constr_id);
            } else if (ctype2 == 6){
                constraint_list = PyList_New(n1);
                if (constraint_list ==  NULL)
                    return NULL;
                create_constraints(constraint_list, constraint1, n2, constraint2, -1L, n2+PyLong_AsLong(PyTuple_GetItem(constraint2, 3)), constr_id);
            } else {
                constraint_list = PyList_New(2);
                if (constraint_list ==  NULL)
                    return NULL;
                PyList_SetItem(constraint_list, 0, Py_NewRef(constraint1));
                PyList_SetItem(constraint_list, 1, Py_NewRef(constraint2));
                PyObject *return_this = create_new_variable(2, 1, constraint_list, constr_id);
                Py_DECREF(constraint_list);
                return return_this;
            }
        PyObject *return_this = build_PyCompoundConstraints(1, 1, constraint_list, constr_id);
        Py_DECREF(constraint_list);
        return return_this;
        } else {
            PyObject* constraint_list = PyList_New(2);
            if (constraint_list ==  NULL)
                return NULL;
            PyList_SetItem(constraint_list, 0, Py_NewRef(constraint1));
            PyList_SetItem(constraint_list, 1, Py_NewRef(constraint2));
            PyObject *return_this = create_new_variable(2, 1, constraint_list, constr_id);
            Py_DECREF(constraint_list);
            return return_this;
        }
    } else {
        PyObject* constraint_list = PyList_New(2);
        if (constraint_list ==  NULL)
            return NULL;
        PyList_SetItem(constraint_list, 0, Py_NewRef(constraint1));
        PyList_SetItem(constraint_list, 1, Py_NewRef(constraint2));
        PyObject *return_this = create_new_variable(2, 1, constraint_list, constr_id);
        Py_DECREF(constraint_list);
        return return_this;
    }

}

static PyObject* transform_conjunctions(PyObject* constraints, PyObject* constr_id, bool allow_mc)
{
    PyObject* compressed_compound_constraints = compress_conjunctions(constraints, constr_id);
    PyObject* compressed_constraints = PyTuple_GetItem(compressed_compound_constraints, 2);
    long N = (long) PyList_Size(compressed_constraints);
    if (N < 2){
        return compressed_compound_constraints;
    } else if (allow_mc && N < 3) {
        PyObject *return_this = transform_two_conjunctions(PyList_GetItem(compressed_constraints, 0), PyList_GetItem(compressed_constraints, 1), constr_id, allow_mc);
        Py_DECREF(compressed_compound_constraints);
        return return_this;
    } else {
        PyObject *return_this = create_new_variable(PyLong_AsLong(PyTuple_GetItem(compressed_compound_constraints, 0)), PyLong_AsLong(PyTuple_GetItem(compressed_compound_constraints, 1)), compressed_constraints, constr_id);
        Py_DECREF(compressed_compound_constraints);
        return return_this;
    }
}

static PyObject* ctransform(PyObject* compound_constraint, bool allow_mc){
    long b = PyLong_AsLong(PyTuple_GetItem(compound_constraint, 0));
    long sign = PyLong_AsLong(PyTuple_GetItem(compound_constraint, 1));
    PyObject* constraints = PyTuple_GetItem(compound_constraint, 2);
    PyObject* constr_id = PyTuple_GetItem(compound_constraint, 3);
    long N = (long)PyList_Size(constraints);
    long ctype = acp_type(b, N, sign);
    if (ctype==-1){
        PyErr_SetString(PyExc_ValueError, "ERROR: Found contradicition");
        return NULL;
    } else if (ctype==0){
        // No constraint needed
        PyObject* constraint_list = PyList_New(1);
        if (constraint_list ==  NULL)
            return NULL;
        PyList_SetItem(constraint_list, 0, Py_NewRef(Py_None));
        return build_PyCompoundConstraints(0, 1, constraint_list, constr_id);
    } else if (ctype==1){
        return transform_disjunctions(constraints, constr_id, allow_mc);
    }
    else if (ctype==4){
        return transform_conjunctions(constraints, constr_id, allow_mc);
    }
    else
        return create_new_variable(b, sign, constraints, constr_id);
}


static PyObject* transform(PyObject* self, PyObject* args)
{
    PyObject* compound_constraint;
    bool allow_mc;
    if (!PyArg_ParseTuple(args, "Op", &compound_constraint, &allow_mc)){
        return NULL;
    }
    return ctransform(compound_constraint, allow_mc);
}


static PyMethodDef LogicMethods[] = {
        {"transform", transform, METH_VARARGS, "Transform constraints."},
        {NULL, NULL, 0, NULL}
};


static struct PyModuleDef logicfunc = {
    PyModuleDef_HEAD_INIT,
    "logicfunc",
    NULL,
    -1,
    LogicMethods
};

PyMODINIT_FUNC PyInit_logicfunc(void)
{
    PyObject *m;
    m = PyModule_Create(&logicfunc);
    if (!m) {
        printf("Failed to initiate m");
        return NULL;
    }
    return m;
}