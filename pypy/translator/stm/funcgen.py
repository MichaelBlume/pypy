from pypy.rpython.lltypesystem import lltype, rffi
from pypy.objspace.flow.model import Constant
from pypy.translator.c.support import cdecl, c_string_constant
from pypy.translator.stm.rstm import size_of_voidp


def _stm_generic_get(funcgen, op, expr):
    T = funcgen.lltypemap(op.result)
    resulttypename = funcgen.db.gettype(T)
    cresulttypename = cdecl(resulttypename, '')
    newvalue = funcgen.expr(op.result, special_case_void=False)
    #
    assert T is not lltype.Void     # XXX
    fieldsize = rffi.sizeof(T)
    if fieldsize >= size_of_voidp or T == lltype.SingleFloat:
        assert 1      # xxx assert somehow that the field is aligned
        if T == lltype.Float:
            funcname = 'stm_read_double'
        elif T == lltype.SingleFloat:
            funcname = 'stm_read_float'
        elif fieldsize == size_of_voidp:
            funcname = 'stm_read_word'
        elif fieldsize == 8:    # 32-bit only: read a 64-bit field
            funcname = 'stm_read_doubleword'
        else:
            raise NotImplementedError(fieldsize)
        return '%s = (%s)%s((long*)&%s);' % (
            newvalue, cresulttypename, funcname, expr)
    else:
        STRUCT = funcgen.lltypemap(op.args[0]).TO
        if isinstance(STRUCT, lltype.Struct):
            # assume that the object is aligned, and any possible misalignment
            # comes from the field offset, so that it can be resolved at
            # compile-time (by using C macros)
            structdef = funcgen.db.gettypedefnode(STRUCT)
            basename = funcgen.expr(op.args[0])
            fieldname = op.args[1].value
            return '%s = STM_read_partial_word(%s, %s, offsetof(%s, %s));' % (
                newvalue, cresulttypename, basename,
                cdecl(funcgen.db.gettype(STRUCT), ''),
                structdef.c_struct_field_name(fieldname))
        #
        else:
            return '%s = stm_read_partial_word(sizeof(%s), &%s);' % (
                newvalue, cresulttypename, expr)

def _stm_generic_set(funcgen, op, targetexpr, T):
    basename = funcgen.expr(op.args[0])
    newvalue = funcgen.expr(op.args[2], special_case_void=False)
    #
    assert T is not lltype.Void     # XXX
    fieldsize = rffi.sizeof(T)
    if fieldsize >= size_of_voidp or T == lltype.SingleFloat:
        assert 1      # xxx assert somehow that the field is aligned
        if T == lltype.Float:
            funcname = 'stm_write_double'
            newtype = 'double'
        elif T == lltype.SingleFloat:
            funcname = 'stm_write_float'
            newtype = 'float'
        elif fieldsize == size_of_voidp:
            funcname = 'stm_write_word'
            newtype = 'long'
        elif fieldsize == 8:    # 32-bit only: read a 64-bit field
            funcname = 'stm_write_doubleword'
            newtype = 'long long'
        else:
            raise NotImplementedError(fieldsize)
        return '%s((long*)&%s, (%s)%s);' % (
            funcname, targetexpr, newtype, newvalue)
    else:
        itemtypename = funcgen.db.gettype(T)
        citemtypename = cdecl(itemtypename, '')
        return ('stm_write_partial_word(sizeof(%s), &%s, %s);' % (
            citemtypename, targetexpr, newvalue))


def stm_getfield(funcgen, op):
    STRUCT = funcgen.lltypemap(op.args[0]).TO
    structdef = funcgen.db.gettypedefnode(STRUCT)
    baseexpr_is_const = isinstance(op.args[0], Constant)
    expr = structdef.ptr_access_expr(funcgen.expr(op.args[0]),
                                     op.args[1].value,
                                     baseexpr_is_const)
    return _stm_generic_get(funcgen, op, expr)

def stm_setfield(funcgen, op):
    STRUCT = funcgen.lltypemap(op.args[0]).TO
    structdef = funcgen.db.gettypedefnode(STRUCT)
    baseexpr_is_const = isinstance(op.args[0], Constant)
    expr = structdef.ptr_access_expr(funcgen.expr(op.args[0]),
                                     op.args[1].value,
                                     baseexpr_is_const)
    T = op.args[2].concretetype
    return _stm_generic_set(funcgen, op, expr, T)

def stm_getarrayitem(funcgen, op):
    ARRAY = funcgen.lltypemap(op.args[0]).TO
    ptr = funcgen.expr(op.args[0])
    index = funcgen.expr(op.args[1])
    arraydef = funcgen.db.gettypedefnode(ARRAY)
    expr = arraydef.itemindex_access_expr(ptr, index)
    return _stm_generic_get(funcgen, op, expr)

def stm_setarrayitem(funcgen, op):
    ARRAY = funcgen.lltypemap(op.args[0]).TO
    ptr = funcgen.expr(op.args[0])
    index = funcgen.expr(op.args[1])
    arraydef = funcgen.db.gettypedefnode(ARRAY)
    expr = arraydef.itemindex_access_expr(ptr, index)
    T = op.args[2].concretetype
    return _stm_generic_set(funcgen, op, expr, T)


def stm_begin_transaction(funcgen, op):
    return 'STM_begin_transaction();'

def stm_commit_transaction(funcgen, op):
    return 'stm_commit_transaction();'

def stm_begin_inevitable_transaction(funcgen, op):
    return 'stm_begin_inevitable_transaction();'

def stm_declare_variable(funcgen, op):
    # this operation occurs only once at the start of a function if
    # it uses stm_transaction_boundary
    assert funcgen.exception_policy is None
    funcgen.exception_policy = 'stm'
    return 'STM_DECLARE_VARIABLE();'

def stm_transaction_boundary(funcgen, op):
    assert funcgen.exception_policy == 'stm'
    # make code looking like this:
    #
    #     stm_commit_transaction();
    #     {
    #         volatile long tmp_123 = l_123;
    #         setjmp(jmpbuf);
    #         l_123 = tmp_123;
    #     }
    #     stm_begin_transaction(&jmpbuf);
    #
    lines = ['\tsetjmp(jmpbuf);']
    TMPVAR = 'tmp_%s'
    for v in op.args:
        tmpname = TMPVAR % v.name
        cdeclname = cdecl(funcgen.lltypename(v), 'volatile ' + tmpname)
        realname = funcgen.expr(v)
        lines.insert(0, '\t%s = %s;' % (cdeclname, realname))
        lines.append('\t%s = %s;' % (realname, tmpname))
    lines.insert(0, '{')
    lines.insert(0, 'stm_commit_transaction();')
    lines.append('}')
    lines.append('stm_begin_transaction(&jmpbuf);')
    return '\n'.join(lines)

def stm_try_inevitable(funcgen, op):
    info = op.args[0].value
    string_literal = c_string_constant(info)
    return 'stm_try_inevitable(STM_EXPLAIN1(%s));' % (string_literal,)


def op_stm(funcgen, op):
    if not getattr(funcgen.db.translator, 'stm_transformation_applied', None):
        raise AssertionError("STM transformation not applied.  You need '--stm'")
    func = globals()[op.opname]
    return func(funcgen, op)
