#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include <math.h>


static PyMethodDef NpUFuncMethods[] = {
        {NULL, NULL, 0, NULL}
};

static void optimized_bit_allocation_64(char **args, const npy_intp *dimensions,
                            const npy_intp* steps, void* data)
{
    npy_intp i=0;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];
    int64_t grp_size;
    int64_t val_adj = 1;
    int64_t val;
    while(i<n) {
        /*BEGIN main ufunc computation*/
        grp_size = 0;
        val = *(int64_t *)in;
        while(i<n && *(int64_t *)in == val)
        {
            grp_size++;
            *((int64_t *)out) = val_adj;

            i++;
            in += in_step;
            out += out_step;
        }
        // i+=grp_size;
        val_adj = val_adj*grp_size + val_adj;
        /*END main ufunc computation*/
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&optimized_bit_allocation_64};

/* These are the input and return dtypes of prio2weight.*/
static char types[2] = {NPY_INT64, NPY_INT64};

static void *data[1] = {NULL};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "npufunc",
    NULL,
    -1,
    NpUFuncMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_npufunc(void)
{
    PyObject *m, *optimized_bit_allocation_64, *d;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    optimized_bit_allocation_64 = PyUFunc_FromFuncAndData(funcs, data, types, 1, 1, 1,
                                        PyUFunc_None, "optimized_bit_allocation_64",
                                        "optimized_bit_allocation_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "optimized_bit_allocation_64", optimized_bit_allocation_64);
    Py_DECREF(optimized_bit_allocation_64);

    return m;
}