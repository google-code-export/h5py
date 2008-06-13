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
    Provides access to the low-level HDF5 "H5D" dataset interface

    Most H5D calls are unchanged.  Since dataset I/O is done with Numpy objects,
    read and write calls do not require you to explicitly define a datatype;
    the type of the given Numpy array is used instead.

    The py_* family of functions in this module provide a significantly 
    simpler interface.  They should be sufficient for nearly all dataset
    operations from Python.
"""

# Pyrex compile-time imports
from h5s cimport H5S_ALL, H5S_UNLIMITED, H5S_SCALAR, H5S_SIMPLE, \
                    H5Sget_simple_extent_type, H5Sclose, H5Sselect_all, \
                    H5Sget_simple_extent_ndims, H5Sget_select_npoints
from h5t cimport PY_H5Tclose, H5Tget_size
from h5p cimport H5P_DEFAULT, H5Pclose
from numpy cimport import_array, ndarray, PyArray_DATA
from utils cimport  check_numpy_read, check_numpy_write, \
                    convert_tuple, \
                    emalloc, efree

# Runtime imports
import h5
from h5 import DDict
import h5t
import h5s

import_array()

# === Public constants and data structures ====================================

COMPACT     = H5D_COMPACT
CONTIGUOUS  = H5D_CONTIGUOUS
CHUNKED     = H5D_CHUNKED

ALLOC_TIME_DEFAULT  = H5D_ALLOC_TIME_DEFAULT
ALLOC_TIME_LATE     = H5D_ALLOC_TIME_LATE
ALLOC_TIME_EARLY    = H5D_ALLOC_TIME_EARLY
ALLOC_TIME_INCR     = H5D_ALLOC_TIME_INCR

SPACE_STATUS_NOT_ALLOCATED  = H5D_SPACE_STATUS_NOT_ALLOCATED
SPACE_STATUS_PART_ALLOCATED = H5D_SPACE_STATUS_PART_ALLOCATED
SPACE_STATUS_ALLOCATED      = H5D_SPACE_STATUS_ALLOCATED

FILL_TIME_ALLOC = H5D_FILL_TIME_ALLOC
FILL_TIME_NEVER = H5D_FILL_TIME_NEVER
FILL_TIME_IFSET = H5D_FILL_TIME_IFSET

FILL_VALUE_UNDEFINED    = H5D_FILL_VALUE_UNDEFINED
FILL_VALUE_DEFAULT      = H5D_FILL_VALUE_DEFAULT
FILL_VALUE_USER_DEFINED = H5D_FILL_VALUE_USER_DEFINED

# === Basic dataset operations ================================================

def create(int loc_id, char* name, hid_t type_id, hid_t space_id, hid_t plist=H5P_DEFAULT):
    """ ( INT loc_id, STRING name, INT type_id, INT space_id,
          INT plist=H5P_DEFAULT ) 
        => INT dataset_id

        Create a new dataset under an HDF5 file or group id.  Keyword plist 
        should be a dataset creation property list.

        For a friendlier version of this function, try py_create()
    """
    return H5Dcreate(loc_id, name, type_id, space_id, plist)

def open(hid_t loc_id, char* name):
    """ (INT loc_id, STRING name) => INT dataset_id

        Open an existing dataset attached to a group or file object, by name.
    """
    return H5Dopen(loc_id, name)

def close(hid_t dset_id):
    """ (INT dset_id)
    """
    H5Dclose(dset_id)

# === Dataset I/O =============================================================

def read(hid_t dset_id, hid_t mspace_id, hid_t fspace_id, ndarray arr_obj, 
                                                    hid_t plist=H5P_DEFAULT):
    """ ( INT dset_id, INT mspace_id, INT fspace_id, NDARRAY arr_obj, 
          INT plist=H5P_DEFAULT)

        Read data from an HDF5 dataset into a Numpy array.  For maximum 
        flexibility, you can specify dataspaces for the file and the Numpy
        object. Keyword plist may be a dataset transfer property list.

        The provided Numpy array must be writable, C-contiguous, and own
        its data.  If this is not the case, ValueError will be raised and the 
        read will fail.

        It is your responsibility to ensure that the memory dataspace
        provided is compatible with the shape of the Numpy array.  Since a
        wide variety of dataspace configurations are possible, this is not
        checked.  You can easily crash Python by reading in data from too
        large a dataspace.
        
        For a friendlier version of this function, try py_read_slab().
    """
    cdef hid_t mtype_id
    mtype_id = 0

    try:
        mtype_id = h5t.py_translate_dtype(arr_obj.dtype)
        check_numpy_write(arr_obj, -1)

        H5Dread(dset_id, mtype_id, mspace_id, fspace_id, plist, PyArray_DATA(arr_obj))

    finally:
        if mtype_id:
            PY_H5Tclose(mtype_id)
        
def write(hid_t dset_id, hid_t mspace_id, hid_t fspace_id, ndarray arr_obj, 
                                                    hid_t plist=H5P_DEFAULT):
    """ ( INT dset_id, INT mspace_id, INT fspace_id, NDARRAY arr_obj, 
          INT plist=H5P_DEFAULT )

        Write data from a Numpy array to an HDF5 dataset. Keyword plist may be 
        a dataset transfer property list.

        The provided Numpy array must be C-contiguous, and own its data.  If 
        this is not the case, ValueError will be raised and the read will fail.

        For a friendlier version of this function, try py_write_slab()
    """
    cdef hid_t mtype_id
    mtype_id = 0

    try:
        mtype_id = h5t.py_translate_dtype(arr_obj.dtype)
        check_numpy_read(arr_obj, -1)

        H5Dwrite(dset_id, mtype_id, mspace_id, fspace_id, plist, PyArray_DATA(arr_obj))

    finally:
        if mtype_id:
            PY_H5Tclose(mtype_id)

def extend(hid_t dset_id, object shape):
    """ (INT dset_id, TUPLE shape)

        Extend the given dataset so it's at least as big as "shape".  Note that
        a dataset may only be extended up to the maximum dimensions of its
        dataspace, which are fixed when the dataset is created.
    """
    cdef hsize_t* dims
    cdef int rank
    cdef hid_t space_id
    space_id = 0
    dims = NULL

    try:
        space_id = H5Dget_space(dset_id)
        rank = H5Sget_simple_extent_ndims(space_id)

        require_tuple(shape, 0, rank, "shape")
        dims = <hsize_t*>emalloc(sizeof(hsize_t)*rank)
        convert_tuple(shape, dims, rank)
        H5Dextend(dset_id, dims)

    finally:
        efree(dims)
        if space_id:
            H5Sclose(space_id)

# === Dataset inspection ======================================================

def get_space(hid_t dset_id):
    """ (INT dset_id) => INT space_id

        Create and return a new copy of the dataspace for this dataset.  
        You're responsible for closing it.
    """
    return H5Dget_space(dset_id)

def get_space_status(hid_t dset_id):
    """ (INT dset_id) => INT space_status_code

        Determine if space has been allocated for a dataset.  
        Return value is one of:
            SPACE_STATUS_NOT_ALLOCATED
            SPACE_STATUS_PART_ALLOCATED
            SPACE_STATUS_ALLOCATED 
    """
    cdef H5D_space_status_t status
    H5Dget_space_status(dset_id, &status)
    return <int>status

def get_type(hid_t dset_id):
    """ (INT dset_id) => INT type_id

        Create and return a new copy of the datatype for this dataset.
        You're responsible for closing it.
    """
    return H5Dget_type(dset_id)

def get_create_plist(hid_t dset_id):
    """ (INT dset_id) => INT property_list_id

        Create a new copy of the dataset creation property list used when this
        dataset was created.  You're responsible for closing it.
    """
    return H5Dget_create_plist(dset_id)

def get_offset(hid_t dset_id):
    """ (INT dset_id) => LONG offset

        Get the offset of this dataset in the file, in bytes.
    """
    return H5Dget_offset(dset_id)

def get_storage_size(hid_t dset_id):
    """ (INT dset_id) => LONG storage_size

        Determine the amount of file space required for a dataset.  Note this
        only counts the space which has actually been allocated; it may even
        be zero.
    """
    return H5Dget_storage_size(dset_id)

# === Python extensions =======================================================

def py_create(hid_t parent_id, char* name, object data=None, object dtype=None,
              object shape=None, object chunks=None, object compression=None,
              object shuffle=False, object fletcher32=False):
    """ ( INT parent_id, STRING name, NDARRAY data=None, DTYPE dtype=None,
          TUPLE shape=None, TUPLE chunks=None, PY_INT compression=None,
          BOOL shuffle=False, BOOL fletcher32=False )
        => INT dataset_id

        Create an HDF5 dataset from Python.  You must supply *either* a Numpy
        array, in which case the dataset will be initialized to its type,
        shape, and contents, *or* both a tuple giving the dimensions and a 
        Numpy dtype object.

        This function also works for scalar arrays; providing a "shape" tuple 
        of () or a 0-dimensional array for "data" will result in a scalar 
        (h5s.SCALAR) dataspace for the new dataset, rather than a slab
        (h5s.SIMPLE).

        Additional options (* is default):
        chunks          A tuple containing chunk sizes, or *None
        compression     Enable DEFLATE compression at this level (0-9) or *None
        shuffle         Enable/*disable shuffle filter
        fletcher32      Enable/*disable Fletcher32 error detection
    """
    cdef hid_t dset_id
    cdef hid_t type_id
    cdef hid_t space_id
    cdef hid_t plist
    space_id = 0
    type_id = 0
    dset_id = 0
    plist = 0

    if (data is None and not (dtype and shape)) or (data is not None and (dtype or shape)):
        raise ValueError("*Either* a Numpy array *or* both a dtype and shape must be provided.")

    if data is not None:
        shape = data.shape
        dtype = data.dtype

    try:
        if len(shape) == 0:
            space_id = h5s.create(H5S_SCALAR)  # let's be explicit
        else:
            space_id = h5s.create_simple(shape)

        type_id = h5t.py_translate_dtype(dtype)
    
        if( chunks or compression or shuffle or fletcher32):
            plist = h5p.create(H5P_DATASET_CREATE)
            if chunks:
                h5p.set_chunk(plist, chunks)    # required for compression
            if shuffle:
                h5p.set_shuffle(plist)          # must immediately precede compression
            if compression:
                h5p.set_deflate(plist, compression)
            if fletcher32:
                h5p.set_fletcher32(plist)
        else:
            plist = H5P_DEFAULT

        dset_id = create(parent_id, name, type_id, space_id, plist)

        if data is not None:
            write(dset_id, H5S_ALL, H5S_ALL, data)

    finally:
        if space_id:
            H5Sclose(space_id)
        if type_id:
            PY_H5Tclose(type_id)
        if plist:
            H5Pclose(plist)

    return dset_id

def py_read_slab(hid_t ds_id, object start, object count, 
                 object stride=None, dtype=None):
    """ (INT ds_id, TUPLE start, TUPLE count, TUPLE stride=None, 
         DTYPE dtype=None)
        => NDARRAY numpy_array_out

        Read a hyperslab from an existing HDF5 dataset, and return it as a
        Numpy array. Dimensions are specified by:

        start:  Tuple of integers indicating the start of the selection.

        count:  Tuple of integers indicating how many elements to read.

        stride: Pitch of the selection.  Data points at <start> are always
                selected.  If None (default), the HDF5 library default of "1" 
                will be used for all axes.

        If a Numpy dtype object is passed in through "dtype", it will be used
        as the type object for the returned array, and the library will attempt
        to convert between datatypes during the read operation.  If no
        automatic conversion path exists, an exception will be raised.

        As is customary when slicing into Numpy array objects, no dimensions 
        with length 1 are present in the returned array.  Additionally, if the
        HDF5 dataset has a scalar dataspace, then only None or empty tuples are
        allowed for start, count and stride, and the returned array will be
        0-dimensional (arr.shape == ()).
    """
    cdef hid_t mem_space
    cdef hid_t file_space
    cdef hid_t type_id
    cdef int rank
    cdef int i

    mem_space  = 0
    file_space = 0
    type_id    = 0

    try:
        # Obtain the Numpy dtype of the array
        if dtype is None:
            type_id = H5Dget_type(ds_id)
            dtype = h5t.py_translate_h5t(type_id)

        file_space = H5Dget_space(ds_id)
        space_type = H5Sget_simple_extent_type(file_space)
        
        if space_type == H5S_SCALAR:

            # This probably indicates a logic error in the caller's program;
            # don't just ignore it.
            for item in (start, count, stride):
                if item is not None and item != ():
                    raise ValueError("For a scalar dataset, start/count/stride must be None or ().")

            arr = ndarray( (), dtype=dtype)
            read(ds_id, H5S_ALL, H5S_ALL, arr)

        elif space_type == H5S_SIMPLE:

            # Attempt hyperslab selection on the dataset file space. 
            # The selection function performs validation of start/count/stride.
            h5s.select_hyperslab(file_space, start, count, stride)

            # Initialize Numpy array; no singlet dimensions allowed.
            npy_count = []
            for i from 0<=i<len(count):
                if count[i] != 0 and count[i] != 1:
                    npy_count.append(count[i])
            npy_count = tuple(npy_count)
            arr = ndarray(npy_count, dtype=dtype)

            mem_space = h5s.create_simple(npy_count)
            read(ds_id, mem_space, file_space, arr)

        else:
            raise NotImplementedError("Dataspace type %d is unsupported" % space_type)

    finally:
        if mem_space:
            H5Sclose(mem_space)
        if file_space:
            H5Sclose(file_space)
        if type_id:
            PY_H5Tclose(type_id)

    return arr

def py_write_slab(hid_t ds_id, ndarray arr, object start, object stride=None):
    """ (INT ds_id, NDARRAY arr_obj, TUPLE start, TUPLE stride=None)

        Write the entire contents of a Numpy array into an HDF5 dataset.
        The size of the given array must fit within the dataspace of the
        HDF5 dataset.

        start:  Tuple of integers giving offset for write.

        stride: Pitch of write in dataset.  The elements of "start" are always
                selected.  If None, the HDF5 library default value "1" will be 
                used for all dimensions.

        The underlying function depends on write access to the data area of the
        Numpy array.  See the caveats in h5d.write.

        Please note that this function does absolutely no array broadcasting;
        if you want to write a (2,3) array to an (N,2,3) or (2,3,N) dataset,
        you'll have to do it yourself from Numpy.
    """
    cdef hid_t mem_space
    cdef hid_t file_space
    cdef int rank

    mem_space  = 0
    file_space = 0

    count = arr_obj.shape

    try:
        file_space = H5Dget_space(ds_id)
        space_type = H5Sget_simple_extent_type(file_space)
        
        if space_type == H5S_SCALAR:

            for item in (start, count, stride):
                if item is not None and item != ():
                    raise ValueError("For a scalar dataset, start/count/stride must be None or ().")
            write(ds_id, H5S_ALL, H5S_ALL, arr)

        elif space_type == H5S_SIMPLE:

            # Attempt hyperslab selection on the dataset file space. 
            # The selection function performs validation of start/count/stride.
            h5s.select_hyperslab(file_space, start, count, stride)
            mem_space = h5s.create_simple(count)

            write(ds_id, mem_space, file_space, arr)

        else:
            raise ValueError("Dataspace type %d is unsupported" % space_type)

    finally:
        if mem_space:
            H5Sclose(mem_space)
        if file_space:
            H5Sclose(file_space)

def py_shape(hid_t dset_id):
    """ (INT dset_id) => TUPLE shape

        Obtain the dataspace of an HDF5 dataset, as a tuple.
    """
    cdef int space_id
    space_id = 0

    try:
        space_id = H5Dget_space(dset_id)
        shape = h5s.get_simple_extent_dims(space_id)
        return shape
    finally:
        if space_id:
            H5Sclose(space_id)

def py_rank(hid_t dset_id):
    """ (INT dset_id) => INT rank

        Obtain the rank of an HDF5 dataset.
    """
    cdef int space_id
    space_id = 0

    try:
        space_id = H5Dget_space(dset_id)
        return H5Sget_simple_extent_ndims(space_id)
    finally:
        if space_id:
            H5Sclose(space_id)

def py_dtype(hid_t dset_id):
    """ (INT dset_id) => DTYPE numpy_dtype

        Get the datatype of an HDF5 dataset, converted to a Numpy dtype.
    """
    cdef hid_t type_id
    type_id = 0

    try:
        type_id = H5Dget_type(dset_id)
        return h5t.py_translate_h5t(type_id)
    finally:
        if type_id:
            PY_H5Tclose(type_id)

def py_patch(hid_t ds_source, hid_t ds_sink, hid_t transfer_space):
    """ (INT ds_source, INT ds_sink, INT transfer_space)

        Transfer selected elements from one dataset to another.  The transfer
        selection must be compatible with both the source and sink datasets, or
        an exception will be raised. 

        This function will allocate a memory buffer large enough to hold the
        entire selection at once.  Looping and memory limitation constraints 
        are the caller's responsibility.
    """
    cdef hid_t source_space 
    cdef hid_t sink_space
    cdef hid_t mem_space
    cdef hid_t source_type
    cdef void* xfer_buf

    cdef hssize_t npoints
    cdef size_t type_size

    source_space = 0    
    sink_space = 0
    mem_space = 0
    source_type = 0
    xfer_buf = NULL

    try:
        source_space = H5Dget_space(ds_source)
        sink_space = H5Dget_space(sink)
        source_type = H5Dget_type(source)

        npoints = H5Sget_select_npoints(space_id)
        type_size = H5Tget_size(source_type)

        mem_space = h5s.create_simple((npoints,))
        H5Sselect_all(mem_space)

        # This assumes that reading into a contiguous buffer and then writing
        # out again to the same selection preserves the arrangement of data
        # elements.  I think this is a reasonable assumption.

        xfer_buf = emalloc(npoints*type_size)

        # Let the HDF5 library do dataspace validation; the worst that can
        # happen is that the write will fail after taking a while to read.

        H5Dread(ds_source, source_type, mem_space, transfer_space, H5P_DEFAULT, xfer_buf)
        H5Dwrite(ds_sink, source_type, mem_space, transfer_space, H5P_DEFAULT, xfer_buf)

    finally:
        efree(xfer_buf)
        if source_space:
            H5Sclose(source_space)
        if sink_space:
            H5Sclose(sink_space)
        if mem_space:
            H5Sclose(mem_space)
        if source_type:
            PY_H5Tclose(source_type)


PY_LAYOUT = DDict({ H5D_COMPACT: 'COMPACT LAYOUT', 
               H5D_CONTIGUOUS: 'CONTIGUOUS LAYOUT',
               H5D_CHUNKED: 'CHUNKED LAYOUT'})
PY_ALLOC_TIME = DDict({ H5D_ALLOC_TIME_DEFAULT: 'DEFAULT ALLOC TIME', 
                        H5D_ALLOC_TIME_LATE:'LATE ALLOC TIME',
                        H5D_ALLOC_TIME_EARLY: 'EARLY ALLOC TIME', 
                        H5D_ALLOC_TIME_INCR: 'INCR ALLOC TIME' })
PY_SPACE_STATUS = DDict({ H5D_SPACE_STATUS_NOT_ALLOCATED: 'SPACE NOT ALLOCATED', 
                    H5D_SPACE_STATUS_PART_ALLOCATED: 'SPACE PARTIALLY ALLOCATED',
                    H5D_SPACE_STATUS_ALLOCATED: 'SPACE FULLY ALLOCATED'})
PY_FILL_TIME = DDict({ H5D_FILL_TIME_ALLOC: 'FILL AT ALLOCATION TIME',
                        H5D_FILL_TIME_NEVER: 'NEVER FILL',
                        H5D_FILL_TIME_IFSET: 'FILL IF SET'})
PY_FILL_VALUE = DDict({H5D_FILL_VALUE_UNDEFINED: 'UNDEFINED FILL VALUE',
                        H5D_FILL_VALUE_DEFAULT: 'DEFAULT FILL VALUE',
                        H5D_FILL_VALUE_USER_DEFINED: 'USER-DEFINED FILL VALUE'})























