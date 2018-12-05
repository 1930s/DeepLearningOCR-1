/*
dl.c

Copyright © Raphael Finkel 2007-2010 raphael@cs.uky.edu

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#include <math.h>
#include <string.h>
#include "ocr.h"

kd_node_t *categorization; // the root
int RTL;

char* callPythonFuncDL(char* filename, char* function, tuple_t args) {
  Py_Initialize();
  // Confirm that the Python interpreter is looking at this folder path
  PyObject *sysmodule = PyImport_ImportModule("sys");
  PyObject *syspath = PyObject_GetAttrString(sysmodule, "path");
  PyList_Append(syspath, PyUnicode_FromString("."));
  Py_DECREF(syspath);
  Py_DECREF(sysmodule);

  // Get references to the "filename" Python file and
  // "function" inside of said file.
  PyObject *mymodule = PyImport_ImportModule(filename);
  if (mymodule == NULL) {
    PyErr_Print();
    exit(1);
  }
  PyObject *myfunc = PyObject_GetAttrString(mymodule, function);
  if (myfunc == NULL) {
    PyErr_Print();
    exit(1);
  }

  // Convert tuple into a Python list
  // fprintf(stdout, "before allocate\n");
  PyObject *built_tuple = PyList_New((Py_ssize_t) TUPLELENGTH);
  int dimension;
  // fprintf(stdout, "before build\n");
  for (dimension = 0; dimension < TUPLELENGTH; dimension += 1) {
    // fprintf(stdout, "inside build %d\n", dimension);
    // printf("%f\n", args[dimension]);
    PyObject *element = Py_BuildValue("f", args[dimension]);
    int ret = PyList_Append(built_tuple, element);
    assert(ret == 0);
    Py_DECREF(element);
  }
  // fprintf(stdout, "after build\n");

  PyObject *maxLength = Py_BuildValue("i", TUPLELENGTH);
  // PyObject *dataName = Py_BuildValue("s", fontFile);
  // PyObject *modelJsonString = Py_BuildValue("s", modelJson);
  // Call the Python function using the arglist and get its result
  PyObject *result = PyObject_CallFunctionObjArgs(myfunc, built_tuple,
    maxLength, model_list, NULL);
  if (result == NULL) {
    PyErr_Print();
    exit(1);
  }
  char* retval = (char*) PyBytes_AsString(result);

  Py_DECREF(result);
  Py_DECREF(maxLength);
  Py_DECREF(built_tuple);
  Py_DECREF(myfunc);
  Py_DECREF(mymodule);
  return retval;
} // callPythonFunc

const char *ocrValueDL(tuple_t tuple) {
  char* retval = (char*) callPythonFuncDL("dl", "ocrValueDL", tuple);
  return retval;
} // ocrValueDL
