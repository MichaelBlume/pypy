from pypy.interpreter.error import OperationError, operationerrfmt
from pypy.interpreter.gateway import unwrap_spec
from pypy.rpython.lltypesystem import lltype, rffi
from pypy.module.micronumpy import interp_dtype
from pypy.objspace.std.strutil import strip_spaces
from pypy.rlib import jit

FLOAT_SIZE = rffi.sizeof(lltype.Float)

def _fromstring_text(space, s, count, sep, length, dtype):
    from pypy.module.micronumpy.interp_numarray import W_NDimArray

    sep_stripped = strip_spaces(sep)
    skip_bad_vals = len(sep_stripped) == 0

    items = []
    num_items = 0
    idx = 0
    
    while (num_items < count or count == -1) and idx < len(s):
        nextidx = s.find(sep, idx)
        if nextidx < 0:
            nextidx = length
        piece = strip_spaces(s[idx:nextidx])
        if len(piece) > 0 or not skip_bad_vals:
            if len(piece) == 0 and not skip_bad_vals:
                val = dtype.itemtype.default_fromstring(space)
            else:
                try:
                    val = dtype.coerce(space, space.wrap(piece))
                except OperationError, e:
                    if not e.match(space, space.w_ValueError):
                        raise
                    gotit = False
                    while not gotit and len(piece) > 0:
                        piece = piece[:-1]
                        try:
                            val = dtype.coerce(space, space.wrap(piece))
                            gotit = True
                        except OperationError, e:
                            if not e.match(space, space.w_ValueError):
                                raise
                    if not gotit:
                        val = dtype.itemtype.default_fromstring(space)
                    nextidx = length
            items.append(val)
            num_items += 1
        idx = nextidx + 1
    
    if count > num_items:
        raise OperationError(space.w_ValueError, space.wrap(
            "string is smaller than requested size"))

    a = W_NDimArray([num_items], dtype=dtype)
    ai = a.create_iter()
    for val in items:
        a.dtype.setitem(a, ai.offset, val)
        ai = ai.next(1)
    
    return space.wrap(a)

def _fromstring_bin(space, s, count, length, dtype):
    from pypy.module.micronumpy.interp_numarray import W_NDimArray
    
    itemsize = dtype.itemtype.get_element_size()
    assert itemsize >= 0
    if count == -1:
        count = length / itemsize
    if length % itemsize != 0:
        raise operationerrfmt(space.w_ValueError,
                              "string length %d not divisable by item size %d",
                              length, itemsize)
    if count * itemsize > length:
        raise OperationError(space.w_ValueError, space.wrap(
            "string is smaller than requested size"))
        
    a = W_NDimArray([count], dtype=dtype)
    fromstring_loop(a, count, dtype, itemsize, s)
    return space.wrap(a)

fromstring_driver = jit.JitDriver(greens=[], reds=['count', 'itemsize', 'dtype',
                                                   'ai', 'a', 's'])

def fromstring_loop(a, count, dtype, itemsize, s):
    ai = a.create_iter()
    i = 0
    while i < count:
        fromstring_driver.jit_merge_point(a=a, count=count, dtype=dtype,
                                          itemsize=itemsize, s=s, ai=ai)
        start = i*itemsize
        assert start >= 0
        val = dtype.itemtype.runpack_str(s[start:start + itemsize])
        a.dtype.setitem(a, ai.offset, val)
        ai = ai.next(1)
        i += 1

@unwrap_spec(s=str, count=int, sep=str)
def fromstring(space, s, w_dtype=None, count=-1, sep=''):
    dtype = space.interp_w(interp_dtype.W_Dtype,
        space.call_function(space.gettypefor(interp_dtype.W_Dtype), w_dtype)
    )
    length = len(s)
    if sep == '':
        return _fromstring_bin(space, s, count, length, dtype)
    else:
        return _fromstring_text(space, s, count, sep, length, dtype)
