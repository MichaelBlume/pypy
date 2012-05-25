from pypy.interpreter.baseobjspace import ObjSpace, W_Root
from pypy.interpreter.error import OperationError, wrap_oserror
from pypy.interpreter.gateway import unwrap_spec
from pypy.rlib.objectmodel import we_are_translated
from pypy.objspace.std.typeobject import MethodCache
from pypy.objspace.std.mapdict import IndexCache
from pypy.rlib import rposix

def internal_repr(space, w_object):
    return space.wrap('%r' % (w_object,))

def isfake(space, w_obj):
    """Return whether the argument is faked (stolen from CPython). This is
    always False after translation."""
    if we_are_translated():
        return space.w_False
    return space.wrap(bool(w_obj.typedef.fakedcpytype))
    #return space.wrap(bool(getattr(w_obj.typedef, 'fakedcpytype', None)))

def interp_pdb(space):
    """Run an interp-level pdb.
    This is not available in translated versions of PyPy."""
    assert not we_are_translated()
    import pdb
    pdb.set_trace()

@unwrap_spec(name=str)
def method_cache_counter(space, name):
    """Return a tuple (method_cache_hits, method_cache_misses) for calls to
    methods with the name."""
    assert space.config.objspace.std.withmethodcachecounter
    cache = space.fromcache(MethodCache)
    return space.newtuple([space.newint(cache.hits.get(name, 0)),
                           space.newint(cache.misses.get(name, 0))])

def reset_method_cache_counter(space):
    """Reset the method cache counter to zero for all method names."""
    assert space.config.objspace.std.withmethodcachecounter
    cache = space.fromcache(MethodCache)
    cache.misses = {}
    cache.hits = {}
    if space.config.objspace.std.withmapdict:
        cache = space.fromcache(IndexCache)
        cache.misses = {}
        cache.hits = {}

@unwrap_spec(name=str)
def mapdict_cache_counter(space, name):
    """Return a tuple (index_cache_hits, index_cache_misses) for lookups
    in the mapdict cache with the given attribute name."""
    assert space.config.objspace.std.withmethodcachecounter
    assert space.config.objspace.std.withmapdict
    cache = space.fromcache(IndexCache)
    return space.newtuple([space.newint(cache.hits.get(name, 0)),
                           space.newint(cache.misses.get(name, 0))])

def builtinify(space, w_func):
    from pypy.interpreter.function import Function, BuiltinFunction
    func = space.interp_w(Function, w_func)
    bltn = BuiltinFunction(func)
    return space.wrap(bltn)

@unwrap_spec(ObjSpace, W_Root, str)
def lookup_special(space, w_obj, meth):
    """Lookup up a special method on an object."""
    if space.is_oldstyle_instance(w_obj):
        w_msg = space.wrap("this doesn't do what you want on old-style classes")
        raise OperationError(space.w_TypeError, w_msg)
    w_descr = space.lookup(w_obj, meth)
    if w_descr is None:
        return space.w_None
    return space.get(w_descr, w_obj)

def do_what_I_mean(space):
    return space.wrap(42)

def list_strategy(space, w_list):
    from pypy.objspace.std.listobject import W_ListObject
    if isinstance(w_list, W_ListObject):
        return space.wrap(w_list.strategy._applevel_repr)
    else:
        w_msg = space.wrap("Can only get the list strategy of a list")
        raise OperationError(space.w_TypeError, w_msg)

@unwrap_spec(fd='c_int')
def validate_fd(space, fd):
    try:
        rposix.validate_fd(fd)
    except OSError, e:
        raise wrap_oserror(space, e)
