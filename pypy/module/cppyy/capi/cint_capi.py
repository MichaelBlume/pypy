import py, os

from pypy.translator.tool.cbuild import ExternalCompilationInfo
from pypy.rpython.lltypesystem import rffi
from pypy.rlib import libffi, rdynload

__all__ = ['identify', 'eci', 'c_load_dictionary']

pkgpath = py.path.local(__file__).dirpath().join(os.pardir)
srcpath = pkgpath.join("src")
incpath = pkgpath.join("include")

if os.environ.get("ROOTSYS"):
    import commands
    (stat, incdir) = commands.getstatusoutput("root-config --incdir")
    if stat != 0:        # presumably Reflex-only
        rootincpath = [os.path.join(os.environ["ROOTSYS"], "include")]
        rootlibpath = [os.path.join(os.environ["ROOTSYS"], "lib64"), os.path.join(os.environ["ROOTSYS"], "lib")]
    else:
        rootincpath = [incdir]
        rootlibpath = commands.getoutput("root-config --libdir").split()
else:
    rootincpath = []
    rootlibpath = []

def identify():
    return 'CINT'

ts_reflect = False
ts_call    = False
ts_memory  = 'auto'
ts_helper  = 'auto'

# force loading in global mode of core libraries, rather than linking with
# them as PyPy uses various version of dlopen in various places; note that
# this isn't going to fly on Windows (note that locking them in objects and
# calling dlclose in __del__ seems to come too late, so this'll do for now)
with rffi.scoped_str2charp('libCint.so') as ll_libname:
    _cintdll = rdynload.dlopen(ll_libname, rdynload.RTLD_GLOBAL | rdynload.RTLD_NOW)
with rffi.scoped_str2charp('libCore.so') as ll_libname:
    _coredll = rdynload.dlopen(ll_libname, rdynload.RTLD_GLOBAL | rdynload.RTLD_NOW)

eci = ExternalCompilationInfo(
    separate_module_files=[srcpath.join("cintcwrapper.cxx")],
    include_dirs=[incpath] + rootincpath,
    includes=["cintcwrapper.h"],
    library_dirs=rootlibpath,
    link_extra=["-lCore", "-lCint"],
    use_cpp_linker=True,
)

_c_load_dictionary = rffi.llexternal(
    "cppyy_load_dictionary",
    [rffi.CCHARP], rdynload.DLLHANDLE,
    threadsafe=False,
    compilation_info=eci)

def c_load_dictionary(name):
    result = _c_load_dictionary(name)
    if not result:
        err = rdynload.dlerror()
        raise rdynload.DLOpenError(err)
    return libffi.CDLL(name)       # should return handle to already open file
