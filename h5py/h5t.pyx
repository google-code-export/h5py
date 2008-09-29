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
    HDF5 "H5T" data-type API

    This module contains the datatype identifier class TypeID, and its
    subclasses which represent things like integer/float/compound identifiers.
    The majority of the H5T API is presented as methods on these identifiers.

    1. Translating between Numpy dtypes and HDF5 type objects

       All identifier classes have a property "dtype", returning a Numpy
       dtype which as closely as possible matches the HDF5 type.

       The module function py_create is the complement to this property, and
       is the standard way to translate Numpy dtypes to HDF5 type identifiers.
       Unlike the dtype property, HDF5 datatypes returned by this function are
       guaranteed to be binary-compatible with their Numpy dtype counterparts

    2. Complex numbers

       Complex number support has been refactored in this version of h5py.
       HDF5 has no native concept of a complex number.  Numpy complex types
       are translated to two-element compound types with two floating-point
       fields. When a two-element compound type is encountered in a file with
       compatible field names, it is treated as a complex type.

       The names for complex types are set as a property on the global
       configuration object, available at "h5py.config".

    3. Enumerated types

       NumPy has no native concept of an enumerated type.  Data of this type
       will be read from the HDF5 file as integers, depending on the base
       type of the enum.

       You can get at the fields of an enum through the standard HDF5 API
       calls, which are presented as methods of class TypeEnumID.
       Additionally, the py_create function allows you to create HDF5
       enumerated types by passing in a dictionary along with a Numpy dtype.

    4. Variable-length types

       VLEN datatype objects can be manipulated, but reading and writing data
       in VLEN format is not supported.  This applies to VLEN strings as well.

    5. Datatypes can be pickled if HDF5 1.8.X is available.
"""

include "config.pxi"
include "sync.pxi"

# Pyrex compile-time imports
from h5 cimport init_hdf5, H5PYConfig, get_config, PHIL, get_phil
from numpy cimport dtype, ndarray
from python_string cimport PyString_FromStringAndSize

from utils cimport  emalloc, efree, \
                    require_tuple, convert_dims, convert_tuple

# Initialization
init_hdf5()

# Runtime imports
import sys
import h5

cdef H5PYConfig cfg = get_config()
cdef PHIL phil = get_phil()

# === Custom C API ============================================================

cdef TypeID typewrap(hid_t id_):

    cdef H5T_class_t cls
    cls = H5Tget_class(id_)

    if cls == H5T_INTEGER:
        pcls = TypeIntegerID
    elif cls == H5T_FLOAT:
        pcls = TypeFloatID
    elif cls == H5T_TIME:
        pcls = TypeTimeID
    elif cls == H5T_STRING:
        pcls = TypeStringID
    elif cls == H5T_BITFIELD:
        pcls = TypeBitfieldID
    elif cls == H5T_OPAQUE:
        pcls = TypeOpaqueID
    elif cls == H5T_COMPOUND:
        pcls = TypeCompoundID
    elif cls == H5T_REFERENCE:
        pcls = TypeReferenceID
    elif cls == H5T_ENUM:
        pcls = TypeEnumID
    elif cls == H5T_VLEN:
        pcls = TypeVlenID
    elif cls == H5T_ARRAY:
        pcls = TypeArrayID
    else:
        pcls = TypeID

    return pcls(id_)

cdef object lockid(hid_t id_in):
    cdef TypeID tid
    tid = typewrap(id_in)
    tid._locked = 1
    return tid

# === Public constants and data structures ====================================


# Enumeration H5T_class_t
NO_CLASS  = H5T_NO_CLASS
INTEGER   = H5T_INTEGER
FLOAT     = H5T_FLOAT
TIME      = H5T_TIME
STRING    = H5T_STRING
BITFIELD  = H5T_BITFIELD
OPAQUE    = H5T_OPAQUE
COMPOUND  = H5T_COMPOUND
REFERENCE = H5T_REFERENCE
ENUM      = H5T_ENUM
VLEN      = H5T_VLEN
ARRAY     = H5T_ARRAY

# Enumeration H5T_sign_t
SGN_NONE   = H5T_SGN_NONE
SGN_2      = H5T_SGN_2

# Enumeration H5T_order_t
ORDER_LE    = H5T_ORDER_LE
ORDER_BE    = H5T_ORDER_BE
ORDER_VAX   = H5T_ORDER_VAX
ORDER_NONE  = H5T_ORDER_NONE

DIR_DEFAULT = H5T_DIR_DEFAULT
DIR_ASCEND  = H5T_DIR_ASCEND
DIR_DESCEND = H5T_DIR_DESCEND

# Enumeration H5T_str_t
STR_NULLTERM = H5T_STR_NULLTERM
STR_NULLPAD  = H5T_STR_NULLPAD
STR_SPACEPAD = H5T_STR_SPACEPAD

# Enumeration H5T_norm_t
NORM_IMPLIED = H5T_NORM_IMPLIED
NORM_MSBSET = H5T_NORM_MSBSET
NORM_NONE = H5T_NORM_NONE

# Enumeration H5T_cset_t:
CSET_ASCII = H5T_CSET_ASCII

# Enumeration H5T_pad_t:
PAD_ZERO = H5T_PAD_ZERO
PAD_ONE = H5T_PAD_ONE
PAD_BACKGROUND = H5T_PAD_BACKGROUND

if sys.byteorder == "little":    # Custom python addition
    ORDER_NATIVE = H5T_ORDER_LE
else:
    ORDER_NATIVE = H5T_ORDER_BE

# --- Built-in HDF5 datatypes -------------------------------------------------

# IEEE floating-point
IEEE_F32LE = lockid(H5T_IEEE_F32LE)
IEEE_F32BE = lockid(H5T_IEEE_F32BE)
IEEE_F64LE = lockid(H5T_IEEE_F64LE)
IEEE_F64BE = lockid(H5T_IEEE_F64BE)

# Signed 2's complement integer types
STD_I8LE  = lockid(H5T_STD_I8LE)
STD_I16LE = lockid(H5T_STD_I16LE)
STD_I32LE = lockid(H5T_STD_I32LE)
STD_I64LE = lockid(H5T_STD_I64LE)

STD_I8BE  = lockid(H5T_STD_I8BE)
STD_I16BE = lockid(H5T_STD_I16BE)
STD_I32BE = lockid(H5T_STD_I32BE)
STD_I64BE = lockid(H5T_STD_I64BE)

# Unsigned integers
STD_U8LE  = lockid(H5T_STD_U8LE)
STD_U16LE = lockid(H5T_STD_U16LE)
STD_U32LE = lockid(H5T_STD_U32LE)
STD_U64LE = lockid(H5T_STD_U64LE)

STD_U8BE  = lockid(H5T_STD_U8BE)
STD_U16BE = lockid(H5T_STD_U16BE)
STD_U32BE = lockid(H5T_STD_U32BE)
STD_U64BE = lockid(H5T_STD_U64BE)

# Native types by bytesize
NATIVE_INT8 = lockid(H5T_NATIVE_INT8)
NATIVE_UINT8 = lockid(H5T_NATIVE_UINT8)
NATIVE_INT16 = lockid(H5T_NATIVE_INT16)
NATIVE_UINT16 = lockid(H5T_NATIVE_UINT16)
NATIVE_INT32 = lockid(H5T_NATIVE_INT32)
NATIVE_UINT32 = lockid(H5T_NATIVE_UINT32)
NATIVE_INT64 = lockid(H5T_NATIVE_INT64)
NATIVE_UINT64 = lockid(H5T_NATIVE_UINT64)
NATIVE_FLOAT = lockid(H5T_NATIVE_FLOAT)
NATIVE_DOUBLE = lockid(H5T_NATIVE_DOUBLE)

# Unix time types
UNIX_D32LE = lockid(H5T_UNIX_D32LE)
UNIX_D64LE = lockid(H5T_UNIX_D64LE)
UNIX_D32BE = lockid(H5T_UNIX_D32BE)
UNIX_D64BE = lockid(H5T_UNIX_D64BE)

# Reference types
STD_REF_OBJ = lockid(H5T_STD_REF_OBJ)
STD_REF_DSETREG = lockid(H5T_STD_REF_DSETREG)

# Null terminated (C) and Fortran string types
C_S1 = lockid(H5T_C_S1)
FORTRAN_S1 = lockid(H5T_FORTRAN_S1)

# Translation tables for HDF5 -> NumPy dtype conversion
cdef dict _order_map = { H5T_ORDER_NONE: '|', H5T_ORDER_LE: '<', H5T_ORDER_BE: '>'}
cdef dict _sign_map  = { H5T_SGN_NONE: 'u', H5T_SGN_2: 'i' }


# === General datatype operations =============================================

@sync
def create(int classtype, size_t size):
    """ (INT classtype, UINT size) => TypeID
        
        Create a new HDF5 type object.  Legal class values are 
        COMPOUND and OPAQUE.  Use enum_create for enums.
    """

    # HDF5 versions 1.6.X segfault with anything else
    if classtype != H5T_COMPOUND and classtype != H5T_OPAQUE:
        raise ValueError("Class must be COMPOUND or OPAQUE.")

    return typewrap(H5Tcreate(<H5T_class_t>classtype, size))

@sync
def open(ObjectID group not None, char* name):
    """ (ObjectID group, STRING name) => TypeID

        Open a named datatype from a file.
    """
    return typewrap(H5Topen(group.id, name))

@sync
def array_create(TypeID base not None, object dims_tpl):
    """ (TypeID base, TUPLE dimensions) => TypeArrayID

        Create a new array datatype, using and HDF5 parent type and
        dimensions given via a tuple of positive integers.  "Unlimited" 
        dimensions are not allowed.
    """
    cdef hsize_t rank
    cdef hsize_t *dims = NULL

    require_tuple(dims_tpl, 0, -1, "dims_tpl")
    rank = len(dims_tpl)
    dims = <hsize_t*>emalloc(sizeof(hsize_t)*rank)

    try:
        convert_tuple(dims_tpl, dims, rank)
        return TypeArrayID(H5Tarray_create(base.id, rank, dims, NULL))
    finally:
        efree(dims)

@sync
def enum_create(TypeID base not None):
    """ (TypeID base) => TypeID

        Create a new enumerated type based on an (integer) parent type.
    """
    return typewrap(H5Tenum_create(base.id))

@sync
def vlen_create(TypeID base not None):
    """ (TypeID base) => TypeID

        Create a new variable-length datatype, using any HDF5 type as a base.

        Although the Python interface can manipulate these types, there is no
        provision for reading/writing VLEN data.
    """
    return typewrap(H5Tvlen_create(base.id))

IF H5PY_18API:
    @sync
    def decode(char* buf):
        """ (STRING buf) => TypeID

            Unserialize an HDF5 type.  You can also do this with the native
            Python pickling machinery.
        """
        return typewrap(H5Tdecode(<unsigned char*>buf))

# === Base type class =========================================================

cdef class TypeID(ObjectID):

    """
        Base class for type identifiers (implements common operations)

        Hashable: If committed; in HDF5 1.8.X, also if locked
        Equality: Logical H5T comparison
    """

    def __hash__(self):
        if self._hash is None:
            try:
                # Try to use object header first
                return ObjectID.__hash__(self)
            except TypeError:
                # It's a transient type object
                IF H5PY_18API:
                    if self._locked:
                        self._hash = hash(self.encode())
                    else:
                        raise TypeError("Only locked or committed types can be hashed")
                ELSE:
                    raise TypeError("Only committed types can be hashed")

        return self._hash

    def __richcmp__(self, object other, int how):
        cdef bint truthval = 0
        if how != 2 and how != 3:
            return NotImplemented
        if isinstance(other, TypeID):
            truthval = self.equal(other)
        
        if how == 2:
            return truthval
        return not truthval

    def __copy__(self):
        cdef TypeID cpy
        cpy = ObjectID.__copy__(self)
        return cpy

    def __repr__(self):
        try:
            dstr = " "+self.dtype.str
        except Exception:
            dstr = ""
        return ObjectID.__repr__(self)+dstr

    property dtype:
        """ A Numpy-style dtype object representing this object.
        """
        def __get__(self):
            return self.py_dtype()

    cdef object py_dtype(self):
        raise TypeError("No NumPy equivalent for %s exists" % self.__class__.__name__)

    @sync
    def commit(self, ObjectID group not None, char* name):
        """ (ObjectID group, STRING name)

            Commit this (transient) datatype to a named datatype in a file.
        """
        H5Tcommit(group.id, name, self.id)

    @sync
    def committed(self):
        """ () => BOOL is_comitted

            Determine if a given type object is named (T) or transient (F).
        """
        return <bint>(H5Tcommitted(self.id))

    @sync
    def copy(self):
        """ () => TypeID

            Create a copy of this type object.
        """
        return typewrap(H5Tcopy(self.id))

    @sync
    def equal(self, TypeID typeid):
        """ (TypeID typeid) => BOOL

            Logical comparison between datatypes.  Also called by
            Python's "==" operator.
        """
        return <bint>(H5Tequal(self.id, typeid.id))

    @sync
    def lock(self):
        """ ()

            Lock this datatype, which makes it immutable and indestructible.
            Once locked, it can't be unlocked.
        """
        H5Tlock(self.id)
        self._locked = 1

    @sync
    def get_class(self):
        """ () => INT classcode

            Determine the datatype's class code.
        """
        return <int>H5Tget_class(self.id)

    @sync
    def set_size(self, size_t size):
        """ (UINT size)

            Set the total size of the datatype, in bytes.
        """
        H5Tset_size(self.id, size)

    @sync
    def get_size(self):
        """ () => INT size

            Determine the total size of a datatype, in bytes.
        """
        return H5Tget_size(self.id)

    @sync
    def get_super(self):
        """ () => TypeID

            Determine the parent type of an array, enumeration or vlen datatype.
        """
        return typewrap(H5Tget_super(self.id))

    @sync
    def detect_class(self, int classtype):
        """ (INT classtype) => BOOL class_is_present

            Determine if a member of the given class exists in a compound
            datatype.  The search is recursive.
        """
        return <bint>(H5Tdetect_class(self.id, <H5T_class_t>classtype))

    @sync
    def _close(self):
        """ Close this datatype.  If it's locked, nothing happens.

            You shouldn't ordinarily need to call this function; datatype
            objects are automatically closed when they're deallocated.
        """
        if not self._locked:
            H5Tclose(self.id)

    IF H5PY_18API:
        @sync
        def encode(self):
            """ () => STRING

                Serialize an HDF5 type.  Bear in mind you can also use the
                native Python pickle/unpickle machinery to do this.  The
                returned string may contain binary values, including NULLs.
            """
            cdef size_t nalloc = 0
            cdef char* buf = NULL

            H5Tencode(self.id, NULL, &nalloc)
            buf = <char*>emalloc(sizeof(char)*nalloc)
            try:
                H5Tencode(self.id, <unsigned char*>buf, &nalloc)
                pystr = PyString_FromStringAndSize(buf, nalloc)
            finally:
                efree(buf)

            return pystr

    IF H5PY_18API:
        # Enable pickling

        def __reduce__(self):
            return (type(self), (-1,), self.encode())

        @sync
        def __setstate__(self, char* state):
            self.id = H5Tdecode(<unsigned char*>state)


# === Top-level classes (inherit directly from TypeID) ========================

cdef class TypeArrayID(TypeID):

    """
        Represents an array datatype
    """

    @sync
    def get_array_ndims(self):
        """ () => INT rank

            Get the rank of the given array datatype.
        """
        return H5Tget_array_ndims(self.id)

    @sync
    def get_array_dims(self):
        """ () => TUPLE dimensions

            Get the dimensions of the given array datatype as
            a tuple of integers.
        """
        cdef hsize_t rank   
        cdef hsize_t* dims = NULL

        rank = H5Tget_array_dims(self.id, NULL, NULL)
        dims = <hsize_t*>emalloc(sizeof(hsize_t)*rank)
        try:
            H5Tget_array_dims(self.id, dims, NULL)
            return convert_dims(dims, rank)
        finally:
            efree(dims)

    cdef object py_dtype(self):
        # Numpy translation function for array types
        cdef TypeID tmp_type
        tmp_type = self.get_super()

        base_dtype = tmp_type.py_dtype()

        shape = self.get_array_dims()
        return dtype( (base_dtype, shape) )


cdef class TypeOpaqueID(TypeID):

    """
        Represents an opaque type
    """

    @sync
    def set_tag(self, char* tag):
        """ (STRING tag)

            Set a string describing the contents of an opaque datatype.
            Limited to 256 characters.
        """
        H5Tset_tag(self.id, tag)

    @sync
    def get_tag(self):
        """ () => STRING tag

            Get the tag associated with an opaque datatype.
        """
        cdef char* buf = NULL

        try:
            buf = H5Tget_tag(self.id)
            assert buf != NULL
            tag = buf
            return tag
        finally:
            free(buf)

    cdef object py_dtype(self):
        # Numpy translation function for opaque types
        return dtype("|V" + str(self.get_size()))

cdef class TypeStringID(TypeID):

    """
        String datatypes, both fixed and vlen.
    """

    @sync
    def is_variable_str(self):
        """ () => BOOL is_variable

            Determine if the given string datatype is a variable-length string.
            Please note that reading/writing data in this format is impossible;
            only fixed-length strings are currently supported.
        """
        return <bint>(H5Tis_variable_str(self.id))

    @sync
    def get_cset(self):
        """ () => INT character_set

            Retrieve the character set used for a string.
        """
        return <int>H5Tget_cset(self.id)

    @sync
    def set_cset(self, int cset):
        """ (INT character_set)

            Set the character set used for a string.
        """
        H5Tset_cset(self.id, <H5T_cset_t>cset)

    @sync
    def get_strpad(self):
        """ () => INT padding_type

            Get the padding type.  Legal values are:
             STR_NULLTERM:  NULL termination only (C style)
             STR_NULLPAD:   Pad buffer with NULLs
             STR_SPACEPAD:  Pad buffer with spaces (FORTRAN style)
        """
        return <int>H5Tget_strpad(self.id)

    @sync
    def set_strpad(self, int pad):
        """ (INT pad)

            Set the padding type.  Legal values are:
             STR_NULLTERM:  NULL termination only (C style)
             STR_NULLPAD:   Pad buffer with NULLs
             STR_SPACEPAD:  Pad buffer with spaces (FORTRAN style)
        """
        H5Tset_strpad(self.id, <H5T_str_t>pad)

    cdef object py_dtype(self):
        # Numpy translation function for string types
        if self.is_variable_str():
            raise TypeError("Variable-length strings are not supported.")

        return dtype("|S" + str(self.get_size()))

cdef class TypeVlenID(TypeID):

    """
        Non-string vlen datatypes.
    """
    pass

cdef class TypeTimeID(TypeID):

    """
        Unix-style time_t (deprecated)
    """
    pass

cdef class TypeBitfieldID(TypeID):

    """
        HDF5 bitfield type
    """
    pass

cdef class TypeReferenceID(TypeID):

    """
        HDF5 object or region reference
    """
    pass


# === Numeric classes (integers and floats) ===================================

cdef class TypeAtomicID(TypeID):

    """
        Base class for atomic datatypes (float or integer)
    """

    @sync
    def get_order(self):
        """ () => INT order

            Obtain the byte order of the datatype; one of:
             ORDER_LE
             ORDER_BE
        """
        return <int>H5Tget_order(self.id)

    @sync
    def set_order(self, int order):
        """ (INT order)

            Set the byte order of the datatype; must be one of
             ORDER_LE
             ORDER_BE
        """
        H5Tset_order(self.id, <H5T_order_t>order)

    @sync
    def get_precision(self):
        """ () => UINT precision

            Get the number of significant bits (excludes padding).
        """
        return H5Tget_precision(self.id)

    @sync
    def set_precision(self, size_t precision):
        """ (UINT precision)
            
            Set the number of significant bits (excludes padding).
        """
        H5Tset_precision(self.id, precision)

    @sync
    def get_offset(self):
        """ () => INT offset

            Get the offset of the first significant bit.
        """
        return H5Tget_offset(self.id)

    @sync
    def set_offset(self, size_t offset):
        """ (UINT offset)

            Set the offset of the first significant bit.
        """
        H5Tset_offset(self.id, offset)

    @sync
    def get_pad(self):
        """ () => (INT lsb_pad_code, INT msb_pad_code)

            Determine the padding type.  Possible values are:
             PAD_ZERO
             PAD_ONE
             PAD_BACKGROUND
        """
        cdef H5T_pad_t lsb
        cdef H5T_pad_t msb
        H5Tget_pad(self.id, &lsb, &msb)
        return (<int>lsb, <int>msb)

    @sync
    def set_pad(self, int lsb, int msb):
        """ (INT lsb_pad_code, INT msb_pad_code)

            Set the padding type.  Possible values are:
             PAD_ZERO
             PAD_ONE
             PAD_BACKGROUND
        """
        H5Tset_pad(self.id, <H5T_pad_t>lsb, <H5T_pad_t>msb)


cdef class TypeIntegerID(TypeAtomicID):

    """
        Integer atomic datatypes
    """

    @sync
    def get_sign(self):
        """ () => INT sign

            Get the "signedness" of the datatype; one of:
              SGN_NONE:  Unsigned
              SGN_2:     Signed 2's complement
        """
        return <int>H5Tget_sign(self.id)

    @sync
    def set_sign(self, int sign):
        """ (INT sign)

            Set the "signedness" of the datatype; one of:
              SGN_NONE:  Unsigned
              SGN_2:     Signed 2's complement
        """
        H5Tset_sign(self.id, <H5T_sign_t>sign)

    cdef object py_dtype(self):
        # Translation function for integer types
        return dtype( _order_map[self.get_order()] + 
                      _sign_map[self.get_sign()] + str(self.get_size()) )


cdef class TypeFloatID(TypeAtomicID):

    """
        Floating-point atomic datatypes
    """

    @sync
    def get_fields(self):
        """ () => TUPLE field_info

            Get information about floating-point bit fields.  See the HDF5
            docs for a better description.  Tuple has to following members:
             0: UINT spos
             1: UINT epos
             2: UINT esize
             3: UINT mpos
             4: UINT msize
        """
        cdef size_t spos, epos, esize, mpos, msize
        H5Tget_fields(self.id, &spos, &epos, &esize, &mpos, &msize)
        return (spos, epos, esize, mpos, msize)

    @sync
    def set_fields(self, size_t spos, size_t epos, size_t esize, 
                          size_t mpos, size_t msize):
        """ (UINT spos, UINT epos, UINT esize, UINT mpos, UINT msize)

            Set floating-point bit fields.  Refer to the HDF5 docs for
            argument definitions.
        """
        H5Tset_fields(self.id, spos, epos, esize, mpos, msize)

    @sync
    def get_ebias(self):
        """ () => UINT ebias

            Get the exponent bias.
        """
        return H5Tget_ebias(self.id)

    @sync
    def set_ebias(self, size_t ebias):
        """ (UINT ebias)

            Set the exponent bias.
        """
        H5Tset_ebias(self.id, ebias)

    @sync
    def get_norm(self):
        """ () => INT normalization_code

            Get the normalization strategy.  Legal values are:
             NORM_IMPLIED
             NORM_MSBSET
             NORM_NONE
        """
        return <int>H5Tget_norm(self.id)

    @sync
    def set_norm(self, int norm):
        """ (INT normalization_code)

            Set the normalization strategy.  Legal values are:
             NORM_IMPLIED
             NORM_MSBSET
             NORM_NONE
        """
        H5Tset_norm(self.id, <H5T_norm_t>norm)

    @sync
    def get_inpad(self):
        """ () => INT pad_code

            Determine the internal padding strategy.  Legal values are:
             PAD_ZERO
             PAD_ONE
             PAD_BACKGROUND
        """
        return <int>H5Tget_inpad(self.id)

    @sync
    def set_inpad(self, int pad_code):
        """ (INT pad_code)

            Set the internal padding strategy.  Legal values are:
             PAD_ZERO
             PAD_ONE
             PAD_BACKGROUND
        """
        H5Tset_inpad(self.id, <H5T_pad_t>pad_code)

    cdef object py_dtype(self):
        # Translation function for floating-point types
        return dtype( _order_map[self.get_order()] + "f" + \
                      str(self.get_size()) )


# === Composite types (enums and compound) ====================================

cdef class TypeCompositeID(TypeID):

    """
        Base class for enumerated and compound types.
    """

    @sync
    def get_nmembers(self):
        """ () => INT number_of_members

            Determine the number of members in a compound or enumerated type.
        """
        return H5Tget_nmembers(self.id)

    @sync
    def get_member_name(self, int member):
        """ (INT member) => STRING name
        
            Determine the name of a member of a compound or enumerated type,
            identified by its index (0 <= member < nmembers).
        """
        cdef char* name
        name = NULL

        if member < 0:
            raise ValueError("Member index must be non-negative.")

        try:
            name = H5Tget_member_name(self.id, member)
            assert name != NULL
            pyname = name
        finally:
            free(name)

        return pyname

    @sync
    def get_member_index(self, char* name):
        """ (STRING name) => INT index

            Determine the index of a member of a compound or enumerated datatype
            identified by a string name.
        """
        return H5Tget_member_index(self.id, name)


cdef class TypeCompoundID(TypeCompositeID):

    """
        Represents a compound datatype
    """


    @sync
    def get_member_class(self, int member):
        """ (INT member) => INT class

            Determine the datatype class of the member of a compound type,
            identified by its index (0 <= member < nmembers).
        """
        if member < 0:
            raise ValueError("Member index must be non-negative.")
        return H5Tget_member_class(self.id, member)


    @sync
    def get_member_offset(self, int member):
        """ (INT member) => INT offset

            Determine the offset, in bytes, of the beginning of the specified
            member of a compound datatype.
        """
        if member < 0:
            raise ValueError("Member index must be non-negative.")
        return H5Tget_member_offset(self.id, member)

    @sync
    def get_member_type(self, int member):
        """ (INT member) => TypeID

            Create a copy of a member of a compound datatype, identified by its
            index.
        """
        if member < 0:
            raise ValueError("Member index must be non-negative.")
        return typewrap(H5Tget_member_type(self.id, member))

    @sync
    def insert(self, char* name, size_t offset, TypeID field not None):
        """ (STRING name, UINT offset, TypeID field)

            Add a named member datatype to a compound datatype.  The parameter
            offset indicates the offset from the start of the compound datatype,
            in bytes.
        """
        H5Tinsert(self.id, name, offset, field.id)

    @sync
    def pack(self):
        """ ()

            Recursively removes padding (introduced on account of e.g. compiler
            alignment rules) from a compound datatype.
        """
        H5Tpack(self.id)

    cdef object py_dtype(self):

        cdef TypeID tmp_type
        cdef list field_names
        cdef list field_types
        cdef int nfields
        field_names = []
        field_types = []
        nfields = self.get_nmembers()

        # First step: read field names and their Numpy dtypes into 
        # two separate arrays.
        for i from 0 <= i < nfields:
            tmp_type = self.get_member_type(i)
            field_names.append(self.get_member_name(i))
            field_types.append(tmp_type.py_dtype())


        # 1. Check if it should be converted to a complex number
        if len(field_names) == 2                                and \
            tuple(field_names) == (cfg._r_name, cfg._i_name)    and \
            field_types[0] == field_types[1]                    and \
            field_types[0].kind == 'f':

            bstring = field_types[0].str
            blen = int(bstring[2:])
            nstring = bstring[0] + "c" + str(2*blen)
            typeobj = dtype(nstring)

        # 2. Otherwise, read all fields of the compound type, in HDF5 order.
        else:
            typeobj = dtype(zip(field_names, field_types))

        return typeobj

cdef class TypeEnumID(TypeCompositeID):

    """
        Represents an enumerated type
    """

    cdef int enum_convert(self, long long *buf, int reverse) except -1:
        # Convert the long long value in "buf" to the native representation
        # of this (enumerated) type.  Conversion performed in-place.
        # Reverse: false => llong->type; true => type->llong

        cdef hid_t basetype
        cdef H5T_class_t class_code

        class_code = H5Tget_class(self.id)
        if class_code != H5T_ENUM:
            raise ValueError("This type (class %d) is not of class ENUM" % class_code)

        basetype = H5Tget_super(self.id)
        assert basetype > 0

        try:
            if not reverse:
                H5Tconvert(H5T_NATIVE_LLONG, basetype, 1, buf, NULL, H5P_DEFAULT)
            else:
                H5Tconvert(basetype, H5T_NATIVE_LLONG, 1, buf, NULL, H5P_DEFAULT)
        finally:
            H5Tclose(basetype)

    @sync
    def enum_insert(self, char* name, long long value):
        """ (STRING name, INT/LONG value)

            Define a new member of an enumerated type.  The value will be
            automatically converted to the base type defined for this enum.  If
            the conversion results in overflow, the value will be silently 
            clipped.
        """
        cdef long long buf

        buf = value
        self.enum_convert(&buf, 0)
        H5Tenum_insert(self.id, name, &buf)

    @sync
    def enum_nameof(self, long long value):
        """ (LLONG value) => STRING name

            Determine the name associated with the given value.  Due to a
            limitation of the HDF5 library, this can only retrieve names up to
            1023 characters in length.
        """
        cdef herr_t retval
        cdef char name[1024]
        cdef long long buf

        buf = value
        self.enum_convert(&buf, 0)
        retval = H5Tenum_nameof(self.id, &buf, name, 1024)
        assert retval >= 0
        retstring = name
        return retstring

    @sync
    def enum_valueof(self, char* name):
        """ (STRING name) => LONG value

            Get the value associated with an enum name.
        """
        cdef long long buf

        H5Tenum_valueof(self.id, name, &buf)
        self.enum_convert(&buf, 1)
        return buf

    @sync
    def get_member_value(self, int idx):
        """ (UINT index) => LONG value

            Determine the value for the member at the given zero-based index.
        """
        cdef herr_t retval
        cdef hid_t ptype
        cdef long long val
        ptype = 0

        if idx < 0:
            raise ValueError("Index must be non-negative.")

        H5Tget_member_value(self.id, idx, &val)
        self.enum_convert(&val, 1)
        return val

    cdef object py_dtype(self):
        # Translation function for enum types

        cdef TypeID tmp_type
        tmp_type = self.get_super()
        return tmp_type.py_dtype()


# === Translation from NumPy dtypes to HDF5 type objects ======================

# The following series of native-C functions each translate a specific class
# of NumPy dtype into an HDF5 type object.  The result is guaranteed to be
# transient and unlocked.

cdef dict _float_le = {4: H5T_IEEE_F32LE, 8: H5T_IEEE_F64LE}
cdef dict _float_be = {4: H5T_IEEE_F32BE, 8: H5T_IEEE_F64BE}
cdef dict _float_nt = {4: H5T_NATIVE_FLOAT, 8: H5T_NATIVE_DOUBLE}

cdef dict _int_le = {1: H5T_STD_I8LE, 2: H5T_STD_I16LE, 4: H5T_STD_I32LE, 8: H5T_STD_I64LE}
cdef dict _int_be = {1: H5T_STD_I8BE, 2: H5T_STD_I16BE, 4: H5T_STD_I32BE, 8: H5T_STD_I64BE}
cdef dict _int_nt = {1: H5T_NATIVE_INT8, 2: H5T_NATIVE_INT16, 4: H5T_NATIVE_INT32, 8: H5T_NATIVE_INT64}

cdef dict _uint_le = {1: H5T_STD_U8LE, 2: H5T_STD_U16LE, 4: H5T_STD_U32LE, 8: H5T_STD_U64LE}
cdef dict _uint_be = {1: H5T_STD_U8BE, 2: H5T_STD_U16BE, 4: H5T_STD_U32BE, 8: H5T_STD_U64BE}
cdef dict _uint_nt = {1: H5T_NATIVE_UINT8, 2: H5T_NATIVE_UINT16, 4: H5T_NATIVE_UINT32, 8: H5T_NATIVE_UINT64} 

cdef TypeFloatID _c_float(dtype dt):
    # Floats (single and double)
    cdef hid_t tid

    if dt.byteorder == c'<':
        tid =  _float_le[dt.elsize]
    elif dt.byteorder == c'>':
        tid =  _float_be[dt.elsize]
    else:
        tid =  _float_nt[dt.elsize]

    return TypeFloatID(H5Tcopy(tid))

cdef TypeIntegerID _c_int(dtype dt):
    # Integers (ints and uints)
    cdef hid_t tid

    if dt.kind == c'i':
        if dt.byteorder == c'<':
            tid = _int_le[dt.elsize]
        elif dt.byteorder == c'>':
            tid = _int_be[dt.elsize]
        else:
            tid = _int_nt[dt.elsize]
    elif dt.kind == c'u':
        if dt.byteorder == c'<':
            tid = _uint_le[dt.elsize]
        elif dt.byteorder == c'>':
            tid = _uint_be[dt.elsize]
        else:
            tid = _uint_nt[dt.elsize]
    else:
        raise TypeError('Illegal int kind "%s"' % dt.kind)

    return TypeIntegerID(H5Tcopy(tid))

cdef TypeEnumID _c_enum(dtype dt, dict vals):
    # Enums
    cdef TypeIntegerID base
    cdef TypeEnumID out

    base = _c_int(dt)

    out = TypeEnumID(H5Tenum_create(base.id))
    for name in sorted(vals):
        out.enum_insert(name, vals[name])
    return out

cdef TypeArrayID _c_array(dtype dt):
    # Arrays
    cdef dtype base
    cdef TypeID type_base
    cdef tuple shape

    base, shape = dt.subdtype
    type_base = py_create(base)
    return array_create(type_base, shape)

cdef TypeOpaqueID _c_opaque(dtype dt):
    # Opaque
    return TypeOpaqueID(H5Tcreate(H5T_OPAQUE, dt.itemsize))

cdef TypeStringID _c_string(dtype dt):
    # Strings (fixed-length)
    cdef hid_t tid

    tid = H5Tcopy(H5T_C_S1)
    H5Tset_size(tid, dt.itemsize)
    return TypeStringID(tid)

cdef TypeCompoundID _c_complex(dtype dt):
    # Complex numbers (names depend on cfg)
    global cfg

    cdef hid_t tid, tid_sub
    cdef size_t size, off_r, off_i

    cdef size_t length = dt.itemsize
    cdef char byteorder = dt.byteorder

    if length == 8:
        size = h5py_size_n64
        off_r = h5py_offset_n64_real
        off_i = h5py_offset_n64_imag
        if byteorder == c'<':
            tid_sub = H5T_IEEE_F32LE
        elif byteorder == c'>':
            tid_sub = H5T_IEEE_F32BE
        else:
            tid_sub = H5T_NATIVE_FLOAT
    elif length == 16:
        size = h5py_size_n128
        off_r = h5py_offset_n128_real
        off_i = h5py_offset_n128_imag
        if byteorder == c'<':
            tid_sub = H5T_IEEE_F64LE
        elif byteorder == c'>':
            tid_sub = H5T_IEEE_F64BE
        else:
            tid_sub = H5T_NATIVE_DOUBLE
    else:
        raise TypeError("Illegal length %d for complex dtype" % length)

    tid = H5Tcreate(H5T_COMPOUND, size)
    H5Tinsert(tid, cfg._r_name, off_r, tid_sub)
    H5Tinsert(tid, cfg._i_name, off_i, tid_sub)

    return TypeCompoundID(tid)

cdef TypeCompoundID _c_compound(dtype dt):
    # Compound datatypes

    cdef hid_t tid
    cdef TypeID type_tmp
    cdef dtype dt_tmp
    cdef size_t offset

    cdef dict fields = dt.fields
    cdef tuple names = dt.names

    tid = H5Tcreate(H5T_COMPOUND, dt.itemsize)

    for name in names:
        dt_tmp, offset = dt.fields[name]
        type_tmp = py_create(dt_tmp)
        H5Tinsert(tid, name, offset, type_tmp.id)

    return TypeCompoundID(tid)


cpdef TypeID py_create(object dtype_in, dict enum_vals=None):
    """ (OBJECT dtype_in, DICT enum_vals=None) => TypeID

        Given a Numpy dtype object, generate a byte-for-byte memory-compatible
        HDF5 datatype object.  The result is guaranteed to be transient and
        unlocked.

        Argument dtype_in may be a dtype object, or anything which can be
        converted to a dtype, including strings like '<i4'.

        enum_vals:
            A optional dictionary mapping names to integer values.  If the
            type being converted is an integer (Numpy kind i/u), the resulting 
            HDF5 type will be an enumeration with that base type, and the 
            given values. Ignored for all other types.
    """
    cdef dtype dt = dtype(dtype_in)
    cdef char kind = dt.kind

    phil.acquire()
    try:
        # Float
        if kind == c'f':
            return _c_float(dt)
        
        # Integer
        elif kind == c'u' or kind == c'i':

            if enum_vals is not None:
                return _c_enum(dt, enum_vals)
            else:
                return _c_int(dt)

        # Complex
        elif kind == c'c':
            return _c_complex(dt)

        # Compound
        elif kind == c'V' and dt.names is not None:
            return _c_compound(dt)

        # Array or opaque
        elif kind == c'V':
            if dt.subdtype is not None:
                return _c_array(dt)
            else:
                return _c_opaque(dt)

        # String
        elif kind == c'S':
            return _c_string(dt)

        # Unrecognized
        else:
            raise TypeError("No conversion path for dtype: %s" % repr(dt))
    finally:
        phil.release()



