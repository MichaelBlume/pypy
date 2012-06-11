"""
Implementation of interpreter-level 'sys' routines.
"""
import os
import pypy

# ____________________________________________________________
#

class State: 
    def __init__(self, space): 
        self.space = space 

        self.w_modules = space.newdict(module=True)

        self.w_warnoptions = space.newlist([])
        self.w_argv = space.newlist([])
        self.setinitialpath(space) 

    def setinitialpath(self, space):
        from pypy.module.sys.initpath import compute_stdlib_path
        # Initialize the default path
        pypydir = os.path.dirname(os.path.abspath(pypy.__file__))
        srcdir = os.path.dirname(pypydir)
        path = compute_stdlib_path(self, srcdir)
        self.w_path = space.newlist([space.wrap(p) for p in path])


def get(space):
    return space.fromcache(State)

def pypy_getudir(space):
    """NOT_RPYTHON
    (should be removed from interpleveldefs before translation)"""
    from pypy.tool.udir import udir
    return space.wrap(str(udir))

