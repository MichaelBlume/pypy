import math

from pypy.module.micronumpy import interp_dtype, signature
from pypy.rlib import rfloat
from pypy.tool.sourcetools import func_with_new_name


def ufunc(func=None, promote_to_float=False):
    if func is None:
        return lambda func: ufunc(func, promote_to_float)
    call_sig = signature.Call1(func)
    def impl(space, w_obj):
        from pypy.module.micronumpy.interp_numarray import (Call1,
            convert_to_array, Scalar)

        w_obj = convert_to_array(space, w_obj)
        if isinstance(w_obj, Scalar):
            res_dtype = space.fromcache(interp_dtype.W_Float64Dtype)
            return func(res_dtype, w_obj.value).wrap(space)

        new_sig = signature.Signature.find_sig([call_sig, w_obj.signature])
        res_dtype = find_unaryop_result_dtype(space,
            w_obj.find_dtype(),
            promote_to_float=promote_to_float,
        )
        w_res = Call1(new_sig, res_dtype, w_obj)
        w_obj.add_invalidates(w_res)
        return w_res
    return func_with_new_name(impl, "%s_dispatcher" % func.__name__)

def ufunc2(func=None, promote_to_float=False):
    if func is None:
        return lambda func: ufunc2(func, promote_to_float)

    call_sig = signature.Call2(func)
    def impl(space, w_lhs, w_rhs):
        from pypy.module.micronumpy.interp_numarray import (Call2,
            convert_to_array, Scalar)

        w_lhs = convert_to_array(space, w_lhs)
        w_rhs = convert_to_array(space, w_rhs)
        if isinstance(w_lhs, Scalar) and isinstance(w_rhs, Scalar):
            res_dtype = space.fromcache(interp_dtype.W_Float64Dtype)
            return func(res_dtype, w_lhs.value, w_rhs.value).wrap(space)

        new_sig = signature.Signature.find_sig([call_sig, w_lhs.signature, w_rhs.signature])
        res_dtype = find_binop_result_dtype(space,
            w_lhs.find_dtype(), w_rhs.find_dtype(),
            promote_to_float=promote_to_float,
        )
        w_res = Call2(new_sig, res_dtype, w_lhs, w_rhs)
        w_lhs.add_invalidates(w_res)
        w_rhs.add_invalidates(w_res)
        return w_res
    return func_with_new_name(impl, "%s_dispatcher" % func.__name__)

def find_binop_result_dtype(space, dt1, dt2, promote_bools=False, promote_to_float=False):
    # dt1.num should be <= dt2.num
    if dt1.num > dt2.num:
        dt1, dt2 = dt2, dt1
    # Some operations promote op(bool, bool) to return int8, rather than bool
    if promote_bools and (dt1.kind == dt2.kind == interp_dtype.BOOLLTR):
        return space.fromcache(interp_dtype.W_Int8Dtype)
    if promote_to_float:
        return find_unaryop_result_dtype(space, dt2, promote_to_float=True)
    # If they're the same kind, choose the greater one.
    if dt1.kind == dt2.kind:
        return dt2

    # Everything promotes to float, and bool promotes to everything.
    if dt2.kind == interp_dtype.FLOATINGLTR or dt1.kind == interp_dtype.BOOLLTR:
        return dt2

    assert False

def find_unaryop_result_dtype(space, dt, promote_to_float=False,
    promote_to_largest=False):
    if promote_to_float:
        for bytes, dtype in interp_dtype.dtypes_by_num_bytes:
            if dtype.kind == interp_dtype.FLOATINGLTR and dtype.num_bytes >= dt.num_bytes:
                return space.fromcache(dtype)
    if promote_to_largest:
        if dt.kind == interp_dtype.BOOLLTR or dt.kind == interp_dtype.SIGNEDLTR:
            return space.fromcache(interp_dtype.W_Int64Dtype)
        elif dt.kind == interp_dtype.FLOATINGLTR:
            return space.fromcache(interp_dtype.W_Float64Dtype)
        else:
            assert False
    return dt


def ufunc_dtype_caller(ufunc_name, op_name, argcount, **kwargs):
    if argcount == 1:
        @ufunc(**kwargs)
        def impl(res_dtype, value):
            return getattr(res_dtype, op_name)(value)
    elif argcount == 2:
        @ufunc2(**kwargs)
        def impl(res_dtype, lvalue, rvalue):
            return getattr(res_dtype, op_name)(lvalue, rvalue)
    return func_with_new_name(impl, ufunc_name)

for ufunc_def in [
    ("add", "add", 2),
    ("subtract", "sub", 2),
    ("multiply", "mul", 2),
    ("divide", "div", 2),
    ("mod", "mod", 2),
    ("power", "pow", 2),

    ("maximum", "max", 2),
    ("minimum", "min", 2),

    ("copysign", "copysign", 2, {"promote_to_float": True}),

    ("positive", "pos", 1),
    ("negative", "neg", 1),
    ("absolute", "abs", 1),
    ("sign", "sign", 1),
    ("reciprocal", "reciprocal", 1),

    ("fabs", "fabs", 1, {"promote_to_float": True}),
    ("floor", "floor", 1, {"promote_to_float": True}),
    ("exp", "exp", 1, {"promote_to_float": True}),

    ("sin", "sin", 1, {"promote_to_float": True}),
    ("cos", "cos", 1, {"promote_to_float": True}),
    ("tan", "tan", 1, {"promote_to_float": True}),
    ("arcsin", "arcsin", 1, {"promote_to_float": True}),
    ("arccos", "arccos", 1, {"promote_to_float": True}),
    ("arctan", "arctan", 1, {"promote_to_float": True}),
]:
    ufunc_name = ufunc_def[0]
    op_name = ufunc_def[1]
    argcount = ufunc_def[2]
    try:
        extra_kwargs = ufunc_def[3]
    except IndexError:
        extra_kwargs = {}

    globals()[ufunc_name] = ufunc_dtype_caller(ufunc_name, op_name, argcount, **extra_kwargs)
