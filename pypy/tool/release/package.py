#!/usr/bin/env python
""" A sample script that packages PyPy, provided that it's already built.
It uses 'pypy/translator/goal/pypy-c' and parts of the rest of the working
copy.  Usage:

    package.py root-pypy-dir [name-of-archive] [name-of-pypy-c] [destination-for-tarball] [pypy-c-path]

Usually you would do:   package.py ../../.. pypy-VER-PLATFORM
The output is found in the directory /tmp/usession-YOURNAME/build/.
"""

import autopath
import shutil
import sys
import py
import os
import fnmatch
from pypy.tool.udir import udir

if sys.version_info < (2,6): py.test.skip("requires 2.6 so far")

USE_ZIPFILE_MODULE = sys.platform == 'win32'

STDLIB_VER = "2.7"

def ignore_patterns(*patterns):
    """Function that can be used as copytree() ignore parameter.

    Patterns is a sequence of glob-style patterns
    that are used to exclude files"""
    def _ignore_patterns(path, names):
        ignored_names = []
        for pattern in patterns:
            ignored_names.extend(fnmatch.filter(names, pattern))
        return set(ignored_names)
    return _ignore_patterns

class PyPyCNotFound(Exception):
    pass

def fix_permissions(basedir):
    if sys.platform != 'win32':
        os.system("chmod -R a+rX %s" % basedir)

def package(basedir, name='pypy-nightly', rename_pypy_c='pypy',
            copy_to_dir = None, override_pypy_c = None):
    basedir = py.path.local(basedir)
    if override_pypy_c is None:
        basename = 'pypy-c'
        if sys.platform == 'win32':
            basename += '.exe'
        pypy_c = basedir.join('pypy', 'translator', 'goal', basename)
    else:
        pypy_c = py.path.local(override_pypy_c)
    if not pypy_c.check():
        print pypy_c
        if os.path.isdir(os.path.dirname(str(pypy_c))):
            raise PyPyCNotFound(
                'Please compile pypy first, using translate.py,'
                ' or check that you gave the correct path'
                ' (see docstring for more info)')
        else:
            raise PyPyCNotFound(
                'Bogus path: %r does not exist (see docstring for more info)'
                % (os.path.dirname(str(pypy_c)),))
    if sys.platform == 'win32' and not rename_pypy_c.lower().endswith('.exe'):
        rename_pypy_c += '.exe'
    binaries = [(pypy_c, rename_pypy_c)]
    #
    if sys.platform == 'win32':
        #Don't include a mscvrXX.dll, users should get their own.
        #Instructions are provided on the website.

        # Can't rename a DLL: it is always called 'libpypy-c.dll'
        
        for extra in ['libpypy-c.dll',
                      'libexpat.dll', 'sqlite3.dll', 
                      'libeay32.dll', 'ssleay32.dll']:
            p = pypy_c.dirpath().join(extra)
            if not p.check():
                p = py.path.local.sysfind(extra)
                assert p, "%s not found" % (extra,)
            print "Picking %s" % p
            binaries.append((p, p.basename))
    #
    builddir = udir.ensure("build", dir=True)
    pypydir = builddir.ensure(name, dir=True)
    # Careful: to copy lib_pypy, copying just the svn-tracked files
    # would not be enough: there are also ctypes_config_cache/_*_cache.py.
    shutil.copytree(str(basedir.join('lib-python').join(STDLIB_VER)),
                    str(pypydir.join('lib-python').join(STDLIB_VER)),
                    ignore=ignore_patterns('.svn', 'py', '*.pyc', '*~'))
    shutil.copytree(str(basedir.join('lib_pypy')),
                    str(pypydir.join('lib_pypy')),
                    ignore=ignore_patterns('.svn', 'py', '*.pyc', '*~'))
    for file in ['LICENSE', 'README']:
        shutil.copy(str(basedir.join(file)), str(pypydir))
    pypydir.ensure('include', dir=True)
    if sys.platform == 'win32':
        shutil.copyfile(str(pypy_c.dirpath().join("libpypy-c.lib")),
                        str(pypydir.join('include/python27.lib')))
    # we want to put there all *.h and *.inl from trunk/include
    # and from pypy/_interfaces
    includedir = basedir.join('include')
    headers = includedir.listdir('*.h') + includedir.listdir('*.inl')
    for n in headers:
        shutil.copy(str(n), str(pypydir.join('include')))
    #
    spdir = pypydir.ensure('site-packages', dir=True)
    shutil.copy(str(basedir.join('site-packages', 'README')), str(spdir))
    #
    if sys.platform == 'win32':
        bindir = pypydir
    else:
        bindir = pypydir.join('bin')
        bindir.ensure(dir=True)
    for source, target in binaries:
        archive = bindir.join(target)
        shutil.copy(str(source), str(archive))
    old_dir = os.getcwd()
    fix_permissions(builddir)
    try:
        os.chdir(str(builddir))
        #
        # 'strip' fun: see https://codespeak.net/issue/pypy-dev/issue587
        for source, target in binaries:
            if sys.platform == 'win32':
                pass
            elif sys.platform == 'darwin':
                os.system("strip -x " + str(bindir.join(target)))    # ignore errors
            else:
                os.system("strip " + str(bindir.join(target)))    # ignore errors
        #
        if USE_ZIPFILE_MODULE:
            import zipfile
            archive = str(builddir.join(name + '.zip'))
            zf = zipfile.ZipFile(archive, 'w',
                                 compression=zipfile.ZIP_DEFLATED)
            for (dirpath, dirnames, filenames) in os.walk(name):
                for fnname in filenames:
                    filename = os.path.join(dirpath, fnname)
                    zf.write(filename)
            zf.close()
        else:
            archive = str(builddir.join(name + '.tar.bz2'))
            if sys.platform == 'darwin' or sys.platform.startswith('freebsd'):
                e = os.system('tar --numeric-owner -cvjf ' + archive + " " + name)
            elif sys.platform == 'cygwin':
                e = os.system('tar --owner=Administrator --group=Administrators --numeric-owner -cvjf ' + archive + " " + name)
            else:
                e = os.system('tar --owner=root --group=root --numeric-owner -cvjf ' + archive + " " + name)
            if e:
                raise OSError('"tar" returned exit status %r' % e)
    finally:
        os.chdir(old_dir)
    if copy_to_dir is not None:
        print "Copying %s to %s" % (archive, copy_to_dir)
        shutil.copy(archive, str(copy_to_dir))
    else:
        print "Ready in %s" % (builddir,)
    return builddir # for tests

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print >>sys.stderr, __doc__
        sys.exit(1)
    else:
        package(*sys.argv[1:])
