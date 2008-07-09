#!/usr/bin/env python

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
    Setup script for the h5py package.  

    All commands take the usual distutils options, like --home, etc.  Pyrex is
    not required for installation, but will be invoked if the .c files are
    missing, one of the --pyrex options is used, or if a non-default API 
    version or debug level is requested.

    To build:
    python setup.py build

    To install:
    sudo python setup.py install

    To run the test suite locally (won't install anything):
    python setup.py test

    Additional options (for all modes):
        --pyrex         Have Pyrex recompile changed pyx files.
        --pyrex-only    Have Pyrex recompile changed pyx files, and stop.
        --pyrex-force   Recompile all pyx files, regardless of timestamps.
        --no-pyrex      Don't run Pyrex, no matter what

        --api=<n>       Specifies API version.  Only "16" is currently allowed.
        --debug=<n>     If nonzero, compile in debug mode.  The number is
                        interpreted as a logging-module level number.

    Advanced developer options:
    python setup.py dev [--doc] [--clean] [--readme=<name.html>]
        --doc:      Rebuild HTML documentation (requires epydoc)
        --clean:    Wipe out build/ and Pyrex-created .c, .dep files
        --readme:   Compile the RST readme file into an HTML fragment
"""

# === Global constants ========================================================

NAME = 'h5py'
MIN_PYREX = '0.9.8.4'  # for compile_multiple
MIN_NUMPY = '1.0.3'

# If you have your HDF5 *.h files and libraries somewhere not in /usr or
# /usr/local, add that path here.
custom_include_dirs = []    # = ["/some/other/path", "/an/other/path"]
custom_library_dirs = []

AUTO_HDR = "# This file is automatically generated.  Do not edit."

# === Initial imports and utilities ===========================================

from distutils.cmd import Command
from distutils.errors import DistutilsError, DistutilsExecError
from distutils.core import setup
from distutils.extension import Extension
import os
import sys
import shutil

# Distutils tries to use hard links when building source distributions, which 
# fails under a wide variety of network filesystems under Linux.
delattr(os, 'link') # goodbye!

def fatal(instring, code=1):
    print >> sys.stderr, "Fatal: "+instring
    exit(code)

def warn(instring):
    print >> sys.stderr, "Warning: "+instring

# === Parse command line arguments ============================================

ENABLE_PYREX = False
PYREX_ONLY = False
PYREX_FORCE = False
PYREX_FORCE_OFF = False

API_VERS = (1,6)
DEBUG_LEVEL = 0

for arg in sys.argv[:]:
    if arg == '--pyrex':
        ENABLE_PYREX = True
        sys.argv.remove(arg)
    elif arg == '--pyrex-only':
        ENABLE_PYREX = True
        PYREX_ONLY = True
        sys.argv.remove(arg)
    elif arg == '--pyrex-force':
        ENABLE_PYREX=True
        PYREX_FORCE = True
        sys.argv.remove(arg)
    elif arg == '--no-pyrex':
        PYREX_FORCE_OFF = True
        sys.argv.remove(arg)
    elif arg.find('--api=') == 0:
        api = arg[6:]
        if api == '16':
            API_VERS = (1,6)
        else:
            fatal('Unrecognized API version "%s" (only "16" currently allowed)' % api)
        sys.argv.remove(arg)
    elif arg.find('--debug=') == 0:
        DEBUG_LEVEL = int(arg[8:])
        sys.argv.remove(arg)

if 'sdist' in sys.argv and os.path.exists('MANIFEST'):
    warn("Cleaning up stale MANIFEST file")
    os.remove('MANIFEST')

# === Required imports ========================================================

# Check Python version (2.5 or greater required)
if not (sys.version_info[0] == 2 and sys.version_info[1] >= 5):
    fatal("At least Python 2.5 is required to install h5py")

# Check for Numpy (required)
try:
    import numpy
    if numpy.version.version < MIN_NUMPY:
        fatal("Numpy version %s is out of date (>= %s needed)" % (numpy.version.version, MIN_NUMPY))

except ImportError:
    fatal("Numpy not installed (version >= %s required)" % MIN_NUMPY)

# === Versioning ==============================================================

# 1. Read version from VERSION.txt file into VERSION global
# 2. Write it to h5py/version.py
# 3. Copy README into h5py/version.py as docstring

vers_in = open('VERSION.txt', 'r')
VERSION = vers_in.read().strip()
vers_out = open(os.path.join(NAME,'version.py'),'w')
rdfile = open('README.txt','r')
vers_out.write(AUTO_HDR+'\n')
vers_out.write('"""\nPackage "h5py" extended information\n\n%s"""\nversion = "%s"\n\n' % (rdfile.read(), VERSION))
rdfile.close()
vers_out.close()
vers_in.close()

# === Setup configuration & Pyrex options =====================================

# Pyrex extension modules
pyx_modules = ['h5' , 'h5f', 'h5g', 'h5s', 'h5t', 'h5d',
               'h5a', 'h5p', 'h5z', 'h5i', 'h5r', 'utils']

pyx_src_path = 'h5py'
pyx_extra_src = ['utils_low.c']     # C source files required for Pyrex code
pyx_libraries = ['hdf5']            # Libraries to link into Pyrex code

# Compile-time include and library dirs for Pyrex code
pyx_include = [numpy.get_include()] 
pyx_include.extend(['/usr/include', '/usr/local/include'])
pyx_include.extend(custom_include_dirs)
pyx_library_dirs = ['/usr/lib', '/usr/local/lib']
pyx_library_dirs.extend(custom_library_dirs)

# Additional compiler flags for Pyrex code
pyx_extra_args = ['-Wno-unused', '-Wno-uninitialized', '-DH5_USE_16_API']

extra_link_args = []
extra_compile_args = pyx_extra_args

# Pyrex source files (without extension)
pyrex_sources = [os.path.join(pyx_src_path, x) for x in pyx_modules]

# Check if the conditions.pxi file is up-to-date
cond_path = os.path.join(pyx_src_path, 'conditions.pxi')
cond = \
"""
%s

DEF H5PY_API_MAJ = %d
DEF H5PY_API_MIN = %d
DEF H5PY_DEBUG = %d
DEF H5PY_API = "%d.%d"
""" % (AUTO_HDR, API_VERS[0], API_VERS[1], DEBUG_LEVEL,API_VERS[0], API_VERS[1])

try:
    cond_file = open(cond_path,'r')
    cond_present = cond_file.read()
    cond_file.close()
    if cond_present != cond:
        ENABLE_PYREX = True
except IOError:
    ENABLE_PYREX = True
    cond_present = ""

# If for some reason the .c files are missing, Pyrex is required.
if not all([os.path.exists(x+'.c') for x in pyrex_sources]):
    ENABLE_PYREX = True

if ENABLE_PYREX and not PYREX_FORCE_OFF:
    print "Running Pyrex..."
    try:
        from Pyrex.Compiler.Main import Version

        if Version.version >= MIN_PYREX:
            from Pyrex.Compiler.Main import compile_multiple, CompilationOptions

            # If we regenerate the file every time, Pyrex's timestamp checking
            # is useless.  So only replace it if it's out of date.
            if cond_present != cond:
                print "Replacing conditions file..."
                cond_file = open(cond_path,'w')
                cond_file.write(cond)
                cond_file.close()

            opts = CompilationOptions(verbose=True, timestamps=(not PYREX_FORCE))
            results = compile_multiple( [x+'.pyx' for x in pyrex_sources], opts)

            if results.num_errors != 0:
                fatal("%d Pyrex compilation errors encountered; aborting." % results.num_errors)
            if PYREX_ONLY:
                exit(0)
        else:
            fatal("Old Pyrex version %s detected (min %s)" % (Version.version, MIN_PYREX))

    except ImportError:
        fatal("Pyrex recompilation required, but Pyrex not installed.")
else:
    print "Skipping Pyrex..."

# Create extensions
pyx_extensions = []
for module_name in pyx_modules:
    sources  = [os.path.join(pyx_src_path, module_name) +'.c']
    sources += [os.path.join(pyx_src_path, x) for x in pyx_extra_src]

    pyx_extensions.append(
        Extension( 
            NAME+'.'+module_name,
            sources, 
            include_dirs = pyx_include, 
            libraries = pyx_libraries,
            library_dirs = pyx_library_dirs, 
            extra_compile_args = extra_compile_args,
            extra_link_args = extra_link_args
        )
    )

# === Custom extensions for distutils =========================================

class test(Command):
    description = "Build %s and run unit tests" % NAME
    user_options = []

    def initialize_options(self):
        pass
    def finalize_options(self):
        pass

    def run(self):
        buildobj = self.distribution.get_command_obj('build')
        buildobj.run()
        oldpath = sys.path
        try:
            sys.path = [os.path.abspath(buildobj.build_lib)] + oldpath
            import h5py.tests
            if not h5py.tests.runtests():
                raise DistutilsError("Unit tests failed.")
        finally:
            sys.path = oldpath

class dev(Command):
    description = "Developer commands (--doc, --clean, --readme=<file>)"
    user_options = [('doc','d','Rebuild documentation'),
                    ('readme=','r','Compile HTML file from README.txt'),
                    ('clean', 'c', 'Remove built files and Pyrex temp files.')]
    boolean_options = ['doc']

    def initialize_options(self):
        self.doc = False
        self.readme = False
        self.clean = False

    def finalize_options(self):
        pass

    def run(self):
        if self.clean:
            try:
                shutil.rmtree('build')
            except OSError:
                pass
            fnames = [ x+'.dep' for x in pyrex_sources ] + \
                     [ x+'.c' for x in pyrex_sources ] + \
                     [ 'MANIFEST', os.path.join(pyx_src_path, 'version.py')]

            for name in fnames:
                try:
                    os.remove(name)
                except OSError:
                    pass

        if self.doc:
            buildobj = self.distribution.get_command_obj('build')
            buildobj.run()

            retval = os.spawnlp(os.P_WAIT, 'epydoc', '-q', '--html',
                        '-o', 'docs/', '--config', 'docs.cfg', 
                        os.path.join(buildobj.build_lib, NAME) )
            if retval != 0:
                raise DistutilsExecError("Could not run epydoc to build documentation.")

        if self.readme:
            import docutils.core
            fh = open('README.txt','r')
            parts = docutils.core.publish_parts(fh.read(),writer_name='html')
            fh.close()
            fh = open(self.readme,'w')
            fh.write(parts['body'])
            fh.close()

# Add these to the command class dictionary for setup()
CMD_CLASS = {'dev': dev, 'test': test}


# Run setup
setup(
  name = NAME,
  version = VERSION,
  author = 'Andrew Collette',
  url = 'h5py.alfven.org',
  packages = ['h5py','h5py.tests'],
  package_data = {'h5py': ['*.pyx'],  # so source is available for tracebacks
                  'h5py.tests': ['data/*.hdf5']},
  ext_modules = pyx_extensions,
  requires = ['numpy (>=1.0.3)'],
  provides = ['h5py'],
  cmdclass = CMD_CLASS
)



