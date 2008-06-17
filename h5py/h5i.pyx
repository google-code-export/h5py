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
    Identifier interface for object inspection.
"""

# Pyrex compile-time imports
from utils cimport emalloc, efree

# Runtime imports
import h5
from h5 import DDict

# === Public constants and data structures ====================================

BADID       = H5I_BADID
FILE        = H5I_FILE
GROUP       = H5I_GROUP
DATASPACE   = H5I_DATASPACE
DATASET     = H5I_DATASET
ATTR        = H5I_ATTR
REFERENCE   = H5I_REFERENCE
GENPROP_CLS = H5I_GENPROP_CLS
GENPROP_LST = H5I_GENPROP_LST
DATATYPE    = H5I_DATATYPE

PY_TYPE = DDict({ H5I_BADID: 'BAD ID', H5I_FILE: 'FILE', H5I_GROUP: 'GROUP',
            H5I_DATASET: 'DATASET', H5I_ATTR: 'ATTRIBUTE', 
            H5I_REFERENCE: 'REFERENCE', H5I_GENPROP_CLS: 'PROPERTY LIST CLASS',
            H5I_GENPROP_LST: 'PROPERTY LIST', H5I_DATATYPE: 'DATATYPE' })

# === Identifier API ==========================================================

def get_type(ObjectID obj not None):
    """ (ObjectID obj) => INT type_code

        Determine the HDF5 typecode of an arbitrary HDF5 object.  The return 
        value is always one of the type constants defined in this module; if 
        the ID is invalid, BADID is returned.
    """
    return <int>H5Iget_type(obj.id)

def get_name(ObjectID obj not None):
    """ (ObjectID obj) => STRING name or None

        Determine (a) name of an HDF5 object.  Because an object has as many
        names as there are hard links to it, this may not be unique.  If
        the identifier is invalid or is not associated with a name, returns
        None.
    """
    cdef int namelen
    cdef char* name

    namelen = <int>H5Iget_name(obj.id, NULL, 0)
    assert namelen >= 0
    if namelen == 0:
        return None

    name = <char*>emalloc(sizeof(char)*(namelen+1))
    try:
        H5Iget_name(obj.id, name, namelen+1)
        pystring = name
        return pystring
    finally:
        efree(name)

def get_file_id(ObjectID obj not None):
    """ (ObjectID obj) => FileID

        Obtain an identifier for the file in which this object resides,
        re-opening the file if necessary.
    """
    return FileID(H5Iget_file_id(obj.id))

def inc_ref(ObjectID obj not None):
    """ (ObjectID obj)

        Increment the reference count for the given object.

        This function is provided for debugging only.  Reference counting
        is automatically synchronized with Python, and you can easily break
        ObjectID instances by abusing this function.
    """
    H5Iinc_ref(obj.id)

def get_ref(ObjectID obj not None):
    """ (ObjectID obj)

        Retrieve the reference count for the given object.

        This function is provided for debugging only.  Reference counting
        is automatically synchronized with Python, and you can easily break
        ObjectID instances by abusing this function.
    """
    return H5Iget_ref(obj.id)

def dec_ref(ObjectID obj not None):
    """ (ObjectID obj)

        Decrement the reference count for the given object.

        This function is provided for debugging only.  Reference counting
        is automatically synchronized with Python, and you can easily break
        ObjectID instances by abusing this function.
    """
    H5Idec_ref(obj.id)




    











