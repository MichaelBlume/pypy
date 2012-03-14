from pypy.objspace.flow.model import Constant, UnwrapException
from pypy.objspace.flow.operation import OperationName, Arity
from pypy.interpreter.gateway import ApplevelClass
from pypy.interpreter.error import OperationError
from pypy.tool.cache import Cache
from pypy.rlib.rarithmetic import r_uint
from pypy.rlib.objectmodel import we_are_translated
import py

def sc_import(space, fn, args):
    args_w, kwds_w = args.unpack()
    assert kwds_w == {}, "should not call %r with keyword arguments" % (fn,)
    assert len(args_w) > 0 and len(args_w) <= 5, 'import needs 1 to 5 arguments'
    w_name = args_w[0]
    w_None = space.wrap(None)
    w_glob, w_loc, w_frm = w_None, w_None, w_None
    if len(args_w) > 1:
        w_glob = args_w[1]
    if len(args_w) > 2:
        w_loc = args_w[2]
    if len(args_w) > 3:
        w_frm = args_w[3]   
    if not isinstance(w_loc, Constant):
        # import * in a function gives us the locals as Variable
        # we always forbid it as a SyntaxError
        raise SyntaxError, "RPython: import * is not allowed in functions"
    if space.do_imports_immediately:
        name, glob, loc, frm = (space.unwrap(w_name), space.unwrap(w_glob),
                                space.unwrap(w_loc), space.unwrap(w_frm))
        try:
            mod = __import__(name, glob, loc, frm)
        except ImportError, e:
            raise OperationError(space.w_ImportError, space.wrap(str(e)))
        return space.wrap(mod)
    # redirect it, but avoid exposing the globals
    w_glob = Constant({})
    return space.do_operation('simple_call', Constant(__import__),
                               w_name, w_glob, w_loc, w_frm)

def sc_operator(space, fn, args):
    args_w, kwds_w = args.unpack()
    assert kwds_w == {}, "should not call %r with keyword arguments" % (fn,)
    opname = OperationName[fn]
    if len(args_w) != Arity[opname]:
        if opname == 'pow' and len(args_w) == 2:
            args_w = args_w + [Constant(None)]
        elif opname == 'getattr' and len(args_w) == 3:
            return space.do_operation('simple_call', Constant(getattr), *args_w)
        else:
            raise Exception, "should call %r with exactly %d arguments" % (
                fn, Arity[opname])
    # completely replace the call with the underlying
    # operation and its limited implicit exceptions semantic
    return getattr(space, opname)(*args_w)

# _________________________________________________________________________
# a simplified version of the basic printing routines, for RPython programs
class StdOutBuffer:
    linebuf = []
stdoutbuffer = StdOutBuffer()
def rpython_print_item(s):
    buf = stdoutbuffer.linebuf
    for c in s:
        buf.append(c)
    buf.append(' ')
def rpython_print_newline():
    buf = stdoutbuffer.linebuf
    if buf:
        buf[-1] = '\n'
        s = ''.join(buf)
        del buf[:]
    else:
        s = '\n'
    import os
    os.write(1, s)

def sc_applevel(space, app, name, args_w):
    # special case only for print_item and print_newline
    if 'pyopcode' in app.filename and name == 'print_item':
        w_s = space.do_operation('str', *args_w)
        args_w = (w_s,)
    elif 'pyopcode' in app.filename and name == 'print_newline':
        pass
    else:
        raise Exception("not RPython: calling %r from %r" % (name, app))
    func = globals()['rpython_' + name]
    return space.do_operation('simple_call', Constant(func), *args_w)

# _________________________________________________________________________

def sc_r_uint(space, r_uint, args):
    args_w, kwds_w = args.unpack()
    assert not kwds_w
    [w_value] = args_w
    if isinstance(w_value, Constant):
        return Constant(r_uint(w_value.value))
    return space.do_operation('simple_call', space.wrap(r_uint), w_value)

def sc_we_are_translated(space, we_are_translated, args):
    return Constant(True)

def setup(space):
    # fn = pyframe.normalize_exception.get_function(space)
    # this is now routed through the objspace, directly.
    # space.specialcases[fn] = sc_normalize_exception
    space.specialcases[__import__] = sc_import
    # redirect ApplevelClass for print et al.
    space.specialcases[ApplevelClass] = sc_applevel
    # turn calls to built-in functions to the corresponding operation,
    # if possible
    for fn in OperationName:
        space.specialcases[fn] = sc_operator
    # special case to constant-fold r_uint(32-bit-constant)
    # (normally, the 32-bit constant is a long, and is not allowed to
    # show up in the flow graphs at all)
    space.specialcases[r_uint] = sc_r_uint
    # special case we_are_translated() to return True
    space.specialcases[we_are_translated] = sc_we_are_translated
