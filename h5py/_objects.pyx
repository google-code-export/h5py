
"""
    Implements ObjectID base class and global object registry.

    It used to be that we could store the HDF5 identifier in an ObjectID
    and simply close it when the object was deallocated.  However, since
    HDF5 1.8.5 they have started recycling object identifiers, which
    breaks this system.

    We now use a global registry of object identifiers.  This is implemented
    via a dictionary which maps an integer representation of the identifier
    to a weak reference of an ObjectID.  There is only one ObjectID instance
    in the universe for each integer identifier.  When the HDF5 reference
    count for a identifier reaches zero, HDF5 closes the object and reclaims
    the identifier. When this occurs, the identifier and weak reference must
    be deleted from the registry. If an ObjectID is deallocated, it is deleted
    from the registry and the HDF5 reference count is decreased, HDF5 closes
    and reclaims the identifier for future use.

    All interactions with the registry must be synchronized for thread safety.
    You must acquire "registry.lock" before interacting with the registry. The
    registry is not internally synchronized, in the interest of performance: we
    don't want the same thread attempting to acquire the lock multiple times
    during a single operation, if we can avoid it.

    All ObjectIDs and subclasses thereof should be opened with the "open"
    classmethod factory function, such that an existing ObjectID instance can
    be returned from the registry when appropriate.
"""

from defs cimport *

from weakref import ref

# New strategy for identifier recycling management
#
# For objects like SpaceIDs and TypeIDs, close() operations only affect that
# single identifier.  So it's safe simply to set the .id attribute to 0, to
# avoid double-closing in the case of an identifier being recycled.
#
# For objects like FileIDs, a close() can remotely invalidate other objects.
# It's not enough to set the .id of the FileID to zero; if, for example, a
# DatasetID is open, it will be invalidated but still have a nonzero .id, which
# could lead to double-closing when the DatasetID is deallocated.
#
# In the latter case, we have to check all IDs in existence, and zero out their
# identifiers if they're invalid.
#
# Identifiers are tracked by two parallel Python lists.  One holds the ObjectID
# Python identifiers (id(obj)), while the other holds weakref objects which
# point to the ObjectID instances.  Weakrefs are used to ensure that an ObjectID
# which goes out of scope is actually collected.
#
# The basic contract here is that once ._close() is called, the identifier is
# inert and will not participate in double-closing nonsense.  The hard part is
# making sure _close() is always called, in the case of action-at-a-distance
# calls such as H5Fclose in the case of H5P_CLOSE_STRONG.
#
# 1. On initialization:
#       * The hid_t is recorded as .id
#       * The object's Python ID and a weak reference are recorded in parallel
#           lists
#       * The attribute .locked is set to 0 (unlocked)
#       * The attribute .nonlocal_close is set to 0 (local close only)
#       * The attribute .manual_close is set to 0 (do not auto-close on __dealloc__)
#
# 2. When ._close() is called:
#       * If the identifier is locked
#           - If it is invalid, RuntimeError is raised (this should never happen)
#       * The identifier is tested for validity
#           - If the object is valid H5Idec_ref is called.
#       * The attribute .id is set to 0
#       * The Python ID and weakref are removed from the parallel lists 
#       * If .nonlocal_close is nonzero, the lists are swept for invalid
#         ObjectIDs
#           - Any invalid ObjectID is _close()-ed, which removes it from 
#             the lists.
#
# 3. When an object is deallocated:
#       * _close() is called.
#
# Locking:
#   * Whenever the parallel lists are accessed

cdef list reg_refs = []
cdef list reg_ids = []

cdef int reg_add(obj) except -1:
    # Adds an ObjectID to the registry

    reg_ids.append(id(obj))
    reg_refs.append(ref(obj))

cdef int reg_remove(obj) except -1:
    # Removes an ObjectID from the registry

    cdef int idx = reg_ids.index(id(obj))
    del reg_ids[idx]
    del reg_refs[idx]

cdef int sweeping = 0

cdef int reg_sweep() except -1:
    # Goes through the list and deletes all invalid identifiers by
    # manually close()-ing them

    cdef ObjectID obj
    cdef list new_reg_refs = []
    cdef list new_reg_ids = []

    global sweeping, reg_refs, reg_ids

    if sweeping:
        return 0

    sweeping = 1
    try:
        dead_objs = [r() for r in reg_refs if r() is not None and not r().valid]
        for obj in dead_objs:
            obj._c_close()
        reg_refs = [r for r in reg_refs if r() is not None and r().valid]
        reg_ids = [id(r()) for r in reg_refs]
    finally:
        sweeping = 0

cdef class ObjectID:

    """
        Represents an HDF5 identifier.

    """

    property fileno:
        def __get__(self):
            cdef H5G_stat_t stat
            H5Gget_objinfo(self.id, '.', 0, &stat)
            return (stat.fileno[0], stat.fileno[1])

    property valid:
        def __get__(self):
            if not self.id:
                return False
            return H5Iget_type(self.id) > 0

    def __cinit__(self, id):
        self.id = id
        self.locked = 0
        self.nonlocal_close = 0
        self.manual_close = 0
        reg_add(self)

    def _close(self):
        self._c_close()

    cdef int _c_close(self) except -1:
        """ Closes the object and wipes out this instance's hid_t.

        If the instance has .nonlocal_close nonzero, also sweeps the identifier
        list for remotely invalidated ObjectIDs.

        Does nothing in the case of a locked or already closed object
        """
        if self.locked:
            return 0

        if self.id == 0:    # Already closed
            return 0

        if self.valid:
            H5Idec_ref(self.id)

        self.id = 0

        reg_remove(self)
        if self.nonlocal_close:
            reg_sweep()

        return 0

    def __dealloc__(self):
        if not self.manual_close:
            self._c_close()

    def __nonzero__(self):
        return self.valid

    def __copy__(self):
        cdef ObjectID cpy
        cpy = type(self)(self.id)
        return cpy

    def __richcmp__(self, object other, int how):
        """ Default comparison mechanism for HDF5 objects (equal/not-equal)

        Default equality testing:
        1. Objects which are not both ObjectIDs are unequal
        2. Objects with the same HDF5 ID number are always equal
        3. Objects which hash the same are equal
        """
        cdef bint equal = 0

        if how != 2 and how != 3:
            return NotImplemented

        if isinstance(other, ObjectID):
            if self.id == other.id:
                equal = 1
            else:
                try:
                    equal = hash(self) == hash(other)
                except TypeError:
                    pass

        if how == 2:
            return equal
        return not equal

    def __hash__(self):
        """ Default hashing mechanism for HDF5 objects

        Default hashing strategy:
        1. Try to hash based on the object's fileno and objno records
        2. If (1) succeeds, cache the resulting value
        3. If (1) fails, raise TypeError
        """
        cdef H5G_stat_t stat

        if self._hash is None:
            try:
                H5Gget_objinfo(self.id, '.', 0, &stat)
                self._hash = hash((stat.fileno[0], stat.fileno[1], stat.objno[0], stat.objno[1]))
            except Exception:
                raise TypeError("Objects of class %s cannot be hashed" % self.__class__.__name__)

        return self._hash

cdef hid_t pdefault(ObjectID pid):

    if pid is None:
        return <hid_t>H5P_DEFAULT
    return pid.id
