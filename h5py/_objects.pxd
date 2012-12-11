from defs cimport *

cdef class ObjectID:

    cdef object __weakref__
    cdef readonly hid_t id            # Raw HDF5 identifier
    cdef public int locked            # For immortal identifiers (H5Tlock()-ed)
    cdef public int nonlocal_close    # Closing the object may invalidate others
    cdef public int manual_close      # Don't auto-close when deallocated
    cdef object _hash                 # Caches computed hash value

    cdef int _c_close(self)

# Convenience functions
cdef hid_t pdefault(ObjectID pid)

# Inheritance scheme (for top-level cimport and import statements):
#
# _objects, _proxy, h5fd, h5z
# h5i, h5r, utils
# _conv, h5t, h5s
# h5p
# h5d, h5a, h5f, h5g
# h5l

