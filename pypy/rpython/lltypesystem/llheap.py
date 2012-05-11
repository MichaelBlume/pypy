# only for the LLInterpreter.  Don't use directly.

from pypy.rpython.lltypesystem.lltype import pyobjectptr, malloc, free, typeOf
from pypy.rpython.lltypesystem.llmemory import weakref_create, weakref_deref

setfield = setattr
from operator import setitem as setarrayitem
from pypy.rlib.rgc import can_move, collect, add_memory_pressure

def setinterior(toplevelcontainer, inneraddr, INNERTYPE, newvalue,
                offsets=None):
    assert typeOf(newvalue) == INNERTYPE
    # xxx access the address object's ref() directly for performance
    inneraddr.ref()[0] = newvalue

from pypy.rpython.lltypesystem.lltype import cast_ptr_to_int as gc_id

def weakref_create_getlazy(objgetter):
    return weakref_create(objgetter())

def shrink_array(p, smallersize):
    return False


def thread_prepare():
    pass

def thread_run():
    pass

def thread_start():
    pass

def thread_die():
    pass

def pin(obj):
    return False

def unpin(obj):
    raise AssertionError("pin() always returns False, "
                         "so unpin() should not be called")

