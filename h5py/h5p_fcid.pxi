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

cdef class PropFCID(PropCreateID):

    """
        File creation property list.
    """

    @sync
    def get_version(self):
        """() => TUPLE version_info

        Determine version information of various file attributes. 
        Elements are:

        0.  UINT Super block version number
        1.  UINT Freelist version number
        2.  UINT Symbol table version number
        3.  UINT Shared object header version number
        """
        cdef herr_t retval
        cdef unsigned int super_
        cdef unsigned int freelist
        cdef unsigned int stab
        cdef unsigned int shhdr

        H5Pget_version(self.id, &super_, &freelist, &stab, &shhdr)

        return (super_, freelist, stab, shhdr)

    @sync
    def set_userblock(self, hsize_t size):
        """(INT/LONG size)

        Set the file user block size, in bytes.  
        Must be a power of 2, and at least 512.
        """
        H5Pset_userblock(self.id, size)

    @sync
    def get_userblock(self):
        """() => LONG size

        Determine the user block size, in bytes.
        """
        cdef hsize_t size
        H5Pget_userblock(self.id, &size)
        return size

    @sync
    def set_sizes(self, size_t addr, size_t size):
        """(UINT addr, UINT size)

        Set the addressing offsets and lengths for objects 
        in an HDF5 file, in bytes.
        """
        H5Pset_sizes(self.id, addr, size)

    @sync
    def get_sizes(self):
        """() => TUPLE sizes

        Determine addressing offsets and lengths for objects in an 
        HDF5 file, in bytes.  Return value is a 2-tuple with values:

        0.  UINT Address offsets
        1.  UINT Lengths
        """
        cdef size_t addr
        cdef size_t size
        H5Pget_sizes(self.id, &addr, &size)
        return (addr, size)

    @sync
    def set_sym_k(self, unsigned int ik, unsigned int lk):
        """(INT ik, INT lk)

        Symbol table node settings.  See the HDF5 docs for H5Pset_sym_k.
        """
        H5Pset_sym_k(self.id, ik, lk)

    @sync
    def get_sym_k(self):
        """() => TUPLE settings

        Determine symbol table node settings.  See the HDF5 docs for
        H5Pget_sym_k.  Return is a 2-tuple (ik, lk).
        """
        cdef unsigned int ik
        cdef unsigned int lk
        H5Pget_sym_k(self.id, &ik, &lk)
        return (ik, lk)

    @sync
    def set_istore_k(self, unsigned int ik):
        """(UINT ik)

        See hdf5 docs for H5Pset_istore_k.
        """
        H5Pset_istore_k(self.id, ik)
    
    @sync
    def get_istore_k(self):
        """() => UINT ik

        See HDF5 docs for H5Pget_istore_k
        """
        cdef unsigned int ik
        H5Pget_istore_k(self.id, &ik)
        return ik

