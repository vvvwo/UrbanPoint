#include <Python.h>
#include <numpy/arrayobject.h>
#include "calculation.h"
#include <string>

static char module_docstring[] = "This module provides an interface for the slice of a pointcloud";

static char compute_docstring[] = "function slicing a pointcloud";


static PyObject* calculation_compute(PyObject* self, PyObject* args, PyObject* keywds);



static PyMethodDef module_methods[] =
{
	{ "compute", (PyCFunction)calculation_compute, METH_VARARGS | METH_KEYWORDS, compute_docstring },
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef =
{
    PyModuleDef_HEAD_INIT,
    "calculation",     // m_name
    module_docstring,       // m_doc
    -1,                     // m_size
    module_methods,         // m_methods
    NULL,                   // m_reload
    NULL,                   // m_traverse
    NULL,                   // m_clear
    NULL,                   // m_free
};

PyMODINIT_FUNC PyInit_voxel_height(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}

static PyObject* calculation_compute(PyObject* self, PyObject* args, PyObject* keywds)
{
    PyObject* points_obj = NULL;
    float pillar_size = 0;

    static char* kwlist[] = { "points","pillar_size", NULL};


    // Parse the input  
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "Of", kwlist, &points_obj,&pillar_size))
    {
        PyErr_SetString(PyExc_RuntimeError, "Error parsing arguments");
        return NULL;
    }


    PyObject* points_array = PyArray_FROM_OTF(points_obj, NPY_FLOAT, NPY_IN_ARRAY);

    // Verify data was load correctly.
    if (points_array == NULL)
    {
        Py_XDECREF(points_array);
        PyErr_SetString(PyExc_RuntimeError, "Error converting input points to numpy arrays of type float32");
        return NULL;
    }
    // Check that the input array respect the dims
    if ((int)PyArray_NDIM(points_array) != 2 || (int)PyArray_DIM(points_array, 1) != 3)
    {
        Py_XDECREF(points_array);
        PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions : points.shape is not (N, 3)");
        return NULL;
    }

    int N = (int)PyArray_DIM(points_array, 0);

    // Convert PyArray to Cloud C++ class
    vector<PointXYZ> original_points;
    original_points = vector<PointXYZ>((PointXYZ*)PyArray_DATA(points_array), (PointXYZ*)PyArray_DATA(points_array) + N);

    vector<float> voxel_heights;


    calculation(original_points, voxel_heights, pillar_size);

    npy_intp* point_dims = new npy_intp[1];
    point_dims[0] = N;

    PyObject* res = PyArray_SimpleNew(1, point_dims, NPY_FLOAT);

    size_t size_in_bytes = N * sizeof(float);
    memcpy(PyArray_DATA(res), voxel_heights.data(), size_in_bytes);

    Py_DECREF(points_array);

    res = Py_BuildValue("N", res);

    return res;
}