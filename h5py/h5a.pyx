#+
# 
# This file is part of h5py, a low-level Python interface to the HDF5 library.
# 
# Copyright (C) 2008 Andrew Collette
# http://h5py.alfven.org
# License: BSD  (See LICENSE.txt for full license)
# 
# $Date$
# 
#-

"""
    Provides access to the low-level HDF5 "H5A" attribute interface.

    Python extensions:
    py_listattrs(ObjectID)          Get a list of attribute names
    py_exists(ObjectID, STRING)     Test if a named attribute exists
"""

include "std_code.pxi"

# Pyrex compile-time imports
from h5p cimport H5P_DEFAULT
from h5t cimport TypeID, typewrap
from h5s cimport SpaceID, H5Sclose

from numpy cimport import_array, ndarray, PyArray_DATA
from utils cimport  check_numpy_read, check_numpy_write, \
                    emalloc, efree, pybool

# Runtime imports
import h5
import h5t
import h5s

import_array()

# === General attribute operations ============================================

def create(ObjectID loc not None, char* name, TypeID tid not None, 
            SpaceID space not None):
    """ (ObjectID loc, STRING name, TypeID tid, SpaceID space) 
        => INT attr_id

        Create a new attribute attached to a parent object, specifiying an 
        HDF5 datatype and dataspace.
    """
    return AttrID(H5Acreate(loc.id, name, tid.id, space.id, H5P_DEFAULT))

def open_idx(ObjectID loc not None, int idx):
    """ (ObjectID loc_id, UINT idx) => INT attr_id

        Open an exisiting attribute on an object, by zero-based index.
    """
    # If the argument is declared UINT and someone passes in -1, the Pyrex
    # layer happily converts it to something like 2**32 -1, which crashes the
    # HDF5 library.
    if idx < 0:
        raise ValueError("Index must be a non-negative integer.")
    return AttrID(H5Aopen_idx(loc.id, idx))

def open_name(ObjectID loc not None, char* name):
    """ (ObjectID loc, STRING name) => INT attr_id

        Open an existing attribute on an object, by name.
    """
    return AttrID(H5Aopen_name(loc.id, name))

def delete(ObjectID loc not None, char* name):
    """ (ObjectID loc, STRING name)

        Remove an attribute from an object.
    """
    H5Adelete(loc.id, name)

def get_num_attrs(ObjectID loc not None):
    """ (ObjectID loc) => INT number_of_attributes

        Determine the number of attributes attached to an HDF5 object.
    """
    return H5Aget_num_attrs(loc.id)

cdef herr_t iter_cb(hid_t loc_id, char *attr_name, object int_tpl) except -1:
    # Iteration callback.  Returns 0 under normal execution, +1 to stop early
    # if StopIteration is raised, and -1 if any other exception occurrs.
    loc, func, data = int_tpl

    try:
        func(loc, attr_name, data)
    except StopIteration:
        return 1

    return 0

def iterate(ObjectID loc not None, object func, object data=None, int startidx=0):
    """ (ObjectID loc, FUNCTION func, OBJECT data=None, UINT startidx=0)
        => INT last_attribute_index

        Iterate an arbitrary Python function over the attributes attached
        to an object.  You can also start at an arbitrary attribute by
        specifying its (zero-based) index.

        Your function:
        1.  Should accept three arguments: the ObjectID for the parent object, 
            the (STRING) name of the attribute, and an arbitrary Python object
            you provide as data.  Any return value is ignored.
        2.  Raise StopIteration to bail out before all attributes are processed.
        3.  Raising anything else immediately aborts iteration, and the
            exception is propagated.
    """
    cdef unsigned int i
    if startidx < 0:
        raise ValueError("Starting index must be a non-negative integer.")
    i = startidx

    int_tpl = (loc, func, data)

    H5Aiterate(loc.id, &i, <H5A_operator_t>iter_cb, int_tpl)

cdef herr_t list_cb(hid_t loc_id, char *attr_name, object listin):
    
    cdef list thelist
    thelist = listin

    thelist.append(attr_name)
    return 0

def py_listattrs(ObjectID loc not None):
    """ (ObjectID loc) => LIST

        Get a list of the names of the attributes attached to an object.
    """
    cdef list retlist
    cdef unsigned int i
    i = 0
    retlist = []
    H5Aiterate(loc.id, &i, <H5A_operator_t>list_cb, retlist)
    return retlist

cdef herr_t cb_exist(hid_t loc_id, char* attr_name, object ref_name):

    if ref_name == attr_name:
        return 1
    return 0

def py_exists(ObjectID loc not None, object ref_name):
    """ (ObjectID loc, STRING ref_name) => BOOL

        Determine if an attribute named "ref_name" is attached to this object.
    """
    cdef unsigned int i
    cdef herr_t retval
    i=0

    retval = H5Aiterate(loc.id, &i, <H5A_operator_t>cb_exist, ref_name)
    
    return pybool(retval)
        
# === Attribute class & methods ===============================================

cdef class AttrID(ObjectID):

    """
        Logical representation of an HDF5 attribute identifier.

        Objects of this class can be used in any HDF5 function call
        which expects an attribute identifier.  Additionally, all H5A*
        functions which always take an attribute instance as the first
        argument are presented as methods of this class.  

        Properties:

        name:   The attribute's name
        dtype:  A Numpy dtype representing this attribute's type
        shape:  A Numpy-style shape tuple representing the dataspace
    """
    property name:
        """ The attribute's name
        """
        def __get__(self):
            return self.get_name()

    property shape:
        """ A Numpy-style shape tuple representing the attribute's dataspace.
        """
        def __get__(self):

            cdef SpaceID space
            space = self.get_space()
            return space.get_simple_extent_dims()

    property dtype:
        """ A Numpy-stype dtype object representing the attribute's datatype
        """
        def __get__(self):

            cdef TypeID tid
            tid = typewrap(H5Aget_type(self.id))
            return tid.py_dtype()

    def _close(self):
        """ ()

            Close this attribute and release resources.  You don't need to
            call this manually; attributes are automatically destroyed when
            their Python wrappers are freed.
        """
        H5Aclose(self.id)

    def read(self, ndarray arr_obj not None):
        """ (NDARRAY arr_obj)

            Read the attribute data into the given Numpy array.  Note that the 
            Numpy array must have the same shape as the HDF5 attribute, and a 
            conversion-compatible datatype.

            The Numpy array must be writable, C-contiguous and own its data.
            If this is not the case, an ValueError is raised and the read fails.
        """
        cdef TypeID mtype
        cdef hid_t space_id
        space_id = 0

        try:
            space_id = H5Aget_space(self.id)
            check_numpy_write(arr_obj, space_id)

            mtype = h5t.py_create(arr_obj.dtype)

            H5Aread(self.id, mtype.id, PyArray_DATA(arr_obj))

        finally:
            if space_id:
                H5Sclose(space_id)

    def write(self, ndarray arr_obj not None):
        """ (NDARRAY arr_obj)

            Write the contents of a Numpy array too the attribute.  Note that
            the Numpy array must have the same shape as the HDF5 attribute, and
            a conversion-compatible datatype.  

            The Numpy array must be C-contiguous and own its data.  If this is
            not the case, ValueError will be raised and the write will fail.
        """
        cdef TypeID mtype
        cdef hid_t space_id
        space_id = 0

        try:
            space_id = H5Aget_space(self.id)
            check_numpy_read(arr_obj, space_id)
            mtype = h5t.py_create(arr_obj.dtype)

            H5Awrite(self.id, mtype.id, PyArray_DATA(arr_obj))

        finally:
            if space_id:
                H5Sclose(space_id)

    def get_name(self):
        """ () => STRING name

            Determine the name of an attribute, given its identifier.
        """
        cdef int blen
        cdef char* buf
        buf = NULL

        try:
            blen = H5Aget_name(self.id, 0, NULL)
            assert blen >= 0
            buf = <char*>emalloc(sizeof(char)*blen+1)
            blen = H5Aget_name(self.id, blen+1, buf)
            strout = buf
        finally:
            efree(buf)

        return strout

    def get_space(self):
        """ () => INT space_id

            Create and return a copy of the attribute's dataspace.
        """
        return SpaceID(H5Aget_space(self.id))

    def get_type(self):
        """ () => INT type_id

            Create and return a copy of the attribute's datatype.
        """
        return typewrap(H5Aget_type(self.id))











