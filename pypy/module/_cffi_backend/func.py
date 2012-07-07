from pypy.interpreter.error import OperationError, operationerrfmt
from pypy.interpreter.baseobjspace import Wrappable
from pypy.interpreter.gateway import interp2app, unwrap_spec
from pypy.rpython.lltypesystem import lltype, rffi

from pypy.module._cffi_backend import ctypeobj, cdataobj, ctypefunc


# ____________________________________________________________

@unwrap_spec(ctype=ctypeobj.W_CType)
def newp(space, ctype, w_init=None):
    return ctype.newp(w_init)

# ____________________________________________________________

@unwrap_spec(ctype=ctypeobj.W_CType)
def cast(space, ctype, w_ob):
    return ctype.cast(w_ob)

# ____________________________________________________________

@unwrap_spec(ctype=ctypefunc.W_CTypeFunc)
def callback(space, ctype, w_callable, w_error=None):
    from pypy.module._cffi_backend.ccallback import W_CDataCallback
    return W_CDataCallback(space, ctype, w_callable, w_error)

# ____________________________________________________________

@unwrap_spec(cdata=cdataobj.W_CData)
def typeof(space, cdata):
    return cdata.ctype

# ____________________________________________________________

def sizeof(space, w_obj):
    ob = space.interpclass_w(w_obj)
    if isinstance(ob, cdataobj.W_CData):
        size = ob._sizeof()
    elif isinstance(ob, ctypeobj.W_CType):
        size = ob.size
        if size < 0:
            raise operationerrfmt(space.w_ValueError,
                                  "ctype '%s' is of unknown size",
                                  ob.name)
    else:
        raise OperationError(space.w_TypeError,
                            space.wrap("expected a 'cdata' or 'ctype' object"))
    return space.wrap(size)

@unwrap_spec(ctype=ctypeobj.W_CType)
def alignof(space, ctype):
    align = ctype.alignof()
    return space.wrap(align)

@unwrap_spec(ctype=ctypeobj.W_CType, fieldname=str)
def offsetof(space, ctype, fieldname):
    ofs = ctype.offsetof(fieldname)
    return space.wrap(ofs)

@unwrap_spec(ctype=ctypeobj.W_CType)
def _getfields(space, ctype):
    return ctype._getfields()

# ____________________________________________________________

@unwrap_spec(ctype=ctypeobj.W_CType, replace_with=str)
def getcname(space, ctype, replace_with):
    p = ctype.name_position
    s = '%s%s%s' % (ctype.name[:p], replace_with, ctype.name[p:])
    return space.wrap(s)
