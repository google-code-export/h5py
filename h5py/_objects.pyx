
"""
    Implements ObjectID base class and global object registry.
"""

from defs cimport *

from weakref import ref

## {{{ http://code.activestate.com/recipes/577336/ (r3)
from cpython cimport pythread
from cpython.exc cimport PyErr_NoMemory

cdef class FastRLock:
    """Fast, re-entrant locking.

    Under uncongested conditions, the lock is never acquired but only
    counted.  Only when a second thread comes in and notices that the
    lock is needed, it acquires the lock and notifies the first thread
    to release it when it's done.  This is all made possible by the
    wonderful GIL.
    """
    cdef pythread.PyThread_type_lock _real_lock
    cdef long _owner            # ID of thread owning the lock
    cdef int _count             # re-entry count
    cdef int _pending_requests  # number of pending requests for real lock
    cdef bint _is_locked        # whether the real lock is acquired

    def __cinit__(self):
        self._owner = -1
        self._count = 0
        self._is_locked = False
        self._pending_requests = 0
        self._real_lock = pythread.PyThread_allocate_lock()
        if self._real_lock is NULL:
            PyErr_NoMemory()

    def __dealloc__(self):
        if self._real_lock is not NULL:
            pythread.PyThread_free_lock(self._real_lock)
            self._real_lock = NULL

    def acquire(self, bint blocking=True):
        return lock_lock(self, pythread.PyThread_get_thread_ident(), blocking)

    def release(self):
        if self._owner != pythread.PyThread_get_thread_ident():
            raise RuntimeError("cannot release un-acquired lock")
        unlock_lock(self)

    # compatibility with threading.RLock

    def __enter__(self):
        # self.acquire()
        return lock_lock(self, pythread.PyThread_get_thread_ident(), True)

    def __exit__(self, t, v, tb):
        # self.release()
        if self._owner != pythread.PyThread_get_thread_ident():
            raise RuntimeError("cannot release un-acquired lock")
        unlock_lock(self)

    def _is_owned(self):
        return self._owner == pythread.PyThread_get_thread_ident()


cdef inline bint lock_lock(FastRLock lock, long current_thread, bint blocking) nogil:
    # Note that this function *must* hold the GIL when being called.
    # We just use 'nogil' in the signature to make sure that no Python
    # code execution slips in that might free the GIL

    if lock._count:
        # locked! - by myself?
        if current_thread == lock._owner:
            lock._count += 1
            return 1
    elif not lock._pending_requests:
        # not locked, not requested - go!
        lock._owner = current_thread
        lock._count = 1
        return 1
    # need to get the real lock
    return _acquire_lock(
        lock, current_thread,
        pythread.WAIT_LOCK if blocking else pythread.NOWAIT_LOCK)

cdef bint _acquire_lock(FastRLock lock, long current_thread, int wait) nogil:
    # Note that this function *must* hold the GIL when being called.
    # We just use 'nogil' in the signature to make sure that no Python
    # code execution slips in that might free the GIL

    if not lock._is_locked and not lock._pending_requests:
        # someone owns it but didn't acquire the real lock - do that
        # now and tell the owner to release it when done. Note that we
        # do not release the GIL here as we must absolutely be the one
        # who acquires the lock now.
        if not pythread.PyThread_acquire_lock(lock._real_lock, wait):
            return 0
        #assert not lock._is_locked
        lock._is_locked = True
    lock._pending_requests += 1
    with nogil:
        # wait for the lock owning thread to release it
        locked = pythread.PyThread_acquire_lock(lock._real_lock, wait)
    lock._pending_requests -= 1
    #assert not lock._is_locked
    #assert lock._count == 0
    if not locked:
        return 0
    lock._is_locked = True
    lock._owner = current_thread
    lock._count = 1
    return 1

cdef inline void unlock_lock(FastRLock lock) nogil:
    # Note that this function *must* hold the GIL when being called.
    # We just use 'nogil' in the signature to make sure that no Python
    # code execution slips in that might free the GIL

    #assert lock._owner == pythread.PyThread_get_thread_ident()
    #assert lock._count > 0
    lock._count -= 1
    if lock._count == 0:
        lock._owner = -1
        if lock._is_locked:
            pythread.PyThread_release_lock(lock._real_lock)
            lock._is_locked = False
## end of http://code.activestate.com/recipes/577336/ }}}

# New strategy for identifier recycling management
#
# For objects like SpaceIDs and TypeIDs, close() operations only affect that
# single identifier.  So it's safe simply to set the .id attribute to 0, to
# avoid double-closing in the case of an identifier being recycled.
#
# For objects like FileIDs, a close() can remotely invalidate other objects.
# It's not enough to set the .id of the FileID to zero; if, for example, a
# DatasetID is open, it will be invalidated but still have a nonzero .id, which
# could lead to double-closing when it's eventually deallocated.
#
# In the latter case, we have to check all IDs in existence, and zero out their
# identifiers if they're invalid.
#
# Identifiers are tracked by two parallel Python lists.  One holds weakref
# objects which point to the ObjectID instances.  Weakrefs are used to ensure
# that an ObjectID which goes out of scope is actually collected.  The other
# list is an index containing the Python IDs (id(obj)) of the instances.  This
# provides a fast way to remove an object from the registry.
#
# The basic contract here is that once ._close() is called, the identifier is
# inert and will not participate in double-closing nonsense.  The hard part is
# making sure _close() is always called, in the case of action-at-a-distance
# calls such as H5Fclose in the case of H5P_CLOSE_STRONG.
#
# 1. On initialization:
#       * The HDF5 hid_t identifier is recorded as .id
#       * The object's Python ID and a weak reference are recorded
#           in parallel lists
#       * The attribute .locked is set to 0 (unlocked)
#       * The attribute .nonlocal_close is set to 0 (local close only)
#       * The attribute .manual_close is set to 0 (auto-close on __dealloc__)
#
# 2. When ._close() is called:
#       * If the identifier is locked or already closed
#           - Return without doing anything
#       * If the identifier is valid
#           - The HDF5 reference count is decremented (H5Idec_ref)
#       * The attribute .id is set to 0
#       * The Python ID and weakref are removed from the parallel lists 
#       * If .nonlocal_close is nonzero, the lists are swept for
#           invalid ObjectIDs
#           - Any invalid ObjectID is _close()-ed, which removes it from 
#             the lists.
#
# 3. When an object is deallocated:
#       * If .manual_close is not set
#           - The object is _close()-ed
#
# Locking:
#   * Whenever the parallel lists are accessed; on initialization and in
#       the _close() method

cdef FastRLock reglock = FastRLock()

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

cdef int reg_sweep() except -1:
    # Goes through the list and deletes all invalid identifiers by
    # manually close()-ing them

    cdef ObjectID obj
    cdef list dead_objs

    global reg_refs, reg_ids

    dead_objs = [r() for r in reg_refs if r() is not None and not r().valid]
    for obj in dead_objs:
        obj._c_close()
    reg_ids = [id(r()) for r in reg_refs]


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
        global reglock
        self.id = id
        self.locked = 0
        self.nonlocal_close = 0
        self.manual_close = 0
        with reglock:
            reg_add(self)

    def _close(self):
        """ Closes the object and wipes out this instance's hid_t.

        If the instance has .nonlocal_close nonzero, also sweeps the identifier
        list for remotely invalidated ObjectIDs.

        Does nothing in the case of a locked or already closed object
        """
        self._c_close()

    cdef int _c_close(self) except -1:
        # The actual logic is in a C method so it can be called in __dealloc__

        global reglock
        with reglock:

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


# --- Debug/performance functions for identifiers follow ----------------------

def reg_stats():
    """ Return a namedtuple with information about the state of the registry

    0 (total):   Total number of identifiers
    1 (valid):   Number of valid identifiers
    2 (invalid): Invalid entries (ObjectID present, .id nonzero but invalid)
    3 (zero):    Zero entries (ObjectID present, .id zero)
    4 (none):    ObjectID deallocated
    """
    global reglock, reg_refs
    from collections import namedtuple

    n_tot = 0
    n_none = 0
    n_valid = 0
    n_invalid = 0
    n_zero = 0

    with reglock:
        n_tot = len(reg_refs)
        for r in reg_refs:
            o = r()
            if o is None:
                n_none += 1
            elif o.valid:
                n_valid += 1
            elif o.id == 0:
                n_zero += 1
            else:
                n_invalid += 1

    tcls = namedtuple('reg_stats', ['total','valid','invalid','zero','none'])
    return tcls._make((n_tot, n_valid, n_invalid, n_zero, n_none))

def id_stats():
    """ Return a named tuple with information about all live identifiers

    0 (files):      number of live FileIDs
    1 (groups):     number of live GroupIDs
    2 (datasets):   number of live DatasetIDs
    3 (attrs):      number of live AttrIDs
    4 (types):      number of live TypeIDs
    5 (spaces):     number of live SpaceIDs
    6 (proplists):  number of live PropIDs (both instances and list classes)
    7 (other):      unknown instances
    """
    global reglock, reg_refs
    from collections import namedtuple

    from h5py import h5a, h5f, h5g, h5d, h5t, h5s, h5p

    nfiles = 0
    ngroups = 0
    ndsets = 0
    nattrs = 0
    ntypes = 0
    nspaces = 0
    nplists = 0
    nother = 0

    with reglock:
        for r in reg_refs:
            o = r()
            if o is None or not o.valid:
                continue
            if isinstance(o, h5a.AttrID):
                nattrs += 1
            elif isinstance(o, h5f.FileID):
                nfiles += 1
            elif isinstance(o, h5d.DatasetID):
                ndsets += 1
            elif isinstance(o, h5g.GroupID):
                ngroups += 1
            elif isinstance(o, h5t.TypeID):
                ntypes += 1
            elif isinstance(o, h5s.SpaceID):
                nspaces += 1
            elif isinstance(o, h5p.PropID):
                nplists += 1
            else:
                nother += 1

    tcls = namedtuple('id_stats', ['files', 'groups', 'datasets', 'attrs',
                'types', 'spaces', 'proplists','other'])
    return tcls._make((nfiles, ngroups, ndsets, nattrs, ntypes, nspaces,
            nplists, nother))

