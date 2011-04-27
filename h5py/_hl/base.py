import posixpath
import warnings
import os
import sys
import collections

import shared
from h5py import h5i, h5r, h5p, h5f, h5t

def is_hdf5(fname):
    """ Determine if a file is valid HDF5 (False if it doesn't exist). """
    fname = os.path.abspath(fname)

    if os.path.isfile(fname):
        try:
            fname = fname.encode(sys.getfilesystemencoding())
        except (UnicodeError, LookupError):
            pass
        return h5f.is_hdf5(fname)
    return False

def guess_dtype(data):
    """ Attempt to guess an appropriate dtype for the object, returning None
    if nothing is appropriate (or if it should be left up the the array
    constructor to figure out)
    """
    if isinstance(data, h5r.RegionReference):
        return h5t.special_dtype(ref=h5r.RegionReference)
    if isinstance(data, h5r.Reference):
        return h5t.special_dtype(ref=h5r.Reference)
    return None

def checkutf8(encoding):
    a = "".join(x for x in encoding if x.isalnum())
    a = a.lower()
    return a in ('u8', 'utf8')

class SharedConfig():
    pass

filedata = collections.defaultdict(SharedConfig)

    
class HLObject(object):

    """
        Base class for high-level interface objects.
    """

    @property
    def _shared(self):
        """ Settings shared across all objects in an HDF5 file """
        return filedata[self.id.fileno]

    @_shared.deleter
    def _shared(self):
        try:
            del filedata[self.id.fileno]
        except KeyError:
            pass

    def _encode(self, name):
        """ Encode a name according to the current file settings.

        Returns a 2-tuple (encoded name, character set code)

        - Binary strings are always passed as-is, h5t.CSET_ASCII
        - Unicode (non-utf8): encode to given codec, h5t.CSET_ASCII
        - Unicode (utf8 or not specified): encode utf8, h5t.CSET_UTF8

        """
        if isinstance(name, str):
            return name, h5t.CSET_ASCII

        encoding = self._shared.encoding
        if encoding is None:
            encoding = 'utf8'

        name = name.encode(encoding)
        code = h5t.CSET_UTF8 if checkutf8(encoding) else h5t.CSET_ASCII

        return name, code

    def _decode(self, name):
        """ Decode a name according to the current file settings.

        - Try explicit encoding (if set)
        - Try utf-8
        - Return the byte string
        """
        encoding = self._shared.encoding
        ee = ('utf8',) if encoding is None else (encoding, 'utf8')
        for e in ee:
            try:
                return name.decode(e)
            except UnicodeDecodeError:
                pass
        return name

    @property
    def file(self):
        """ Return a File instance associated with this object """
        import files
        return files.File(self)

    @property
    def name(self):
        """ Return the full name of this object.  None if anonymous. """
        return h5i.get_name(self.id)

    @property
    def parent(self):
        """Return the parent group of this object.

        This is always equivalent to file[posixpath.dirname(obj.name)].
        ValueError if this object is anonymous.
        """
        if self.name is None:
            raise ValueError("Parent of an anonymous object is undefined")
        return self.file[posixpath.dirname(self.name)]

    @property
    def id(self):
        """ Low-level identifier appropriate for this object """
        return self._id

    @property
    def ref(self):
        """ An (opaque) HDF5 reference to this object """
        return h5r.create(self.id, '.', h5r.OBJECT)

    @property
    def attrs(self):
        """ Attributes attached to this object """
        import attrs
        return attrs.AttributeManager(self)

    def __init__(self, oid):
        """ Setup this object, given its low-level identifier """
        self._id = oid

    def __nonzero__(self):
        return self.id.__nonzero__()

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if hasattr(other, 'id'):
            return self.id == other.id
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __nonzero__(self):
        return bool(self.id)

class DictCompat(object):

    """
        Contains dictionary-style compatibility methods for groups and
        attributes.
    """
    
    def keys(self):
        """ Get a list containing member names """
        return list(self)

    def iterkeys(self):
        """ Get an iterator over member names """
        return iter(self)

    def values(self):
        """ Get a list containing member objects """
        return [self[x] for x in self]

    def itervalues(self):
        """ Get an iterator over member objects """
        for x in self:
            yield self[x]

    def items(self):
        """ Get a list of tuples containing (name, object) pairs """
        return [(x, self[x]) for x in self]

    def iteritems(self):
        """ Get an iterator over (name, object) pairs """
        for x in self:
            yield (x, self[x])

    def get(self, name, default=None):
        """ Retrieve the member, or return default if it doesn't exist """
        if name in self:
            return self[name]
        return default

    # Compatibility methods
    def listnames(self):
        """ Deprecated alias for keys() """
        warnings.warn("listnames() is deprecated; use keys() instead", DeprecationWarning)
        return self.keys()
    def iternames(self):
        """ Deprecated alias for iterkeys() """
        warnings.warn("iternames() is deprecated; use iterkeys() instead", DeprecationWarning)
        return self.iterkeys()
    def listobjects(self):
        """ Deprecated alias for values() """
        warnings.warn("listobjects() is deprecated; use values() instead", DeprecationWarning)
        return self.values()
    def iterobjects(self):
        """ Deprecated alias for itervalues() """
        warnings.warn("iterobjects() is deprecated; use itervalues() instead", DeprecationWarning)
        return self.itervalues()
    def listitems(self):
        """ Deprecated alias for items() """
        warnings.warn("listitems() is deprecated; use items() instead", DeprecationWarning)
        return self.items()

