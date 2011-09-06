from pypy.rpython.lltypesystem import lltype, rffi
from pypy.jit.backend.llsupport.descr import *
from pypy.jit.backend.llsupport import symbolic
from pypy.rlib.objectmodel import Symbolic
from pypy.rpython.annlowlevel import llhelper
from pypy.jit.metainterp import history
from pypy.jit.codewriter import longlong
import sys, struct, py

def test_get_size_descr():
    c0 = GcCache(False)
    c1 = GcCache(True)
    T = lltype.GcStruct('T')
    S = lltype.GcStruct('S', ('x', lltype.Char),
                             ('y', lltype.Ptr(T)))
    descr_s = get_size_descr(c0, S)
    descr_t = get_size_descr(c0, T)
    assert descr_s.size == symbolic.get_size(S, False)
    assert descr_t.size == symbolic.get_size(T, False)
    assert descr_s.count_fields_if_immutable() == -1
    assert descr_t.count_fields_if_immutable() == -1
    assert descr_s == get_size_descr(c0, S)
    assert descr_s != get_size_descr(c1, S)
    #
    descr_s = get_size_descr(c1, S)
    assert isinstance(descr_s.size, Symbolic)
    assert descr_s.count_fields_if_immutable() == -1

def test_get_size_descr_immut():
    S = lltype.GcStruct('S', hints={'immutable': True})
    T = lltype.GcStruct('T', ('parent', S),
                        ('x', lltype.Char),
                        hints={'immutable': True})
    U = lltype.GcStruct('U', ('parent', T),
                        ('u', lltype.Ptr(T)),
                        ('v', lltype.Signed),
                        hints={'immutable': True})
    V = lltype.GcStruct('V', ('parent', U),
                        ('miss1', lltype.Void),
                        ('miss2', lltype.Void),
                        hints={'immutable': True})
    for STRUCT, expected in [(S, 0), (T, 1), (U, 3), (V, 3)]:
        for translated in [False, True]:
            c0 = GcCache(translated)
            descr_s = get_size_descr(c0, STRUCT)
            assert descr_s.count_fields_if_immutable() == expected

def test_get_field_descr():
    U = lltype.Struct('U')
    T = lltype.GcStruct('T')
    S = lltype.GcStruct('S', ('x', lltype.Char),
                             ('y', lltype.Ptr(T)),
                             ('z', lltype.Ptr(U)),
                             ('f', lltype.Float),
                             ('s', lltype.SingleFloat))
    assert getFieldDescrClass(lltype.Ptr(T)) is GcPtrFieldDescr
    assert getFieldDescrClass(lltype.Ptr(U)) is NonGcPtrFieldDescr
    cls = getFieldDescrClass(lltype.Char)
    assert cls != getFieldDescrClass(lltype.Signed)
    assert cls == getFieldDescrClass(lltype.Char)
    clsf = getFieldDescrClass(lltype.Float)
    assert clsf != cls
    assert clsf == getFieldDescrClass(lltype.Float)
    clss = getFieldDescrClass(lltype.SingleFloat)
    assert clss not in (cls, clsf)
    assert clss == getFieldDescrClass(lltype.SingleFloat)
    assert clss == getFieldDescrClass(rffi.UINT)    # for now
    #
    c0 = GcCache(False)
    c1 = GcCache(True)
    assert get_field_descr(c0, S, 'y') == get_field_descr(c0, S, 'y')
    assert get_field_descr(c0, S, 'y') != get_field_descr(c1, S, 'y')
    for tsc in [False, True]:
        c2 = GcCache(tsc)
        descr_x = get_field_descr(c2, S, 'x')
        descr_y = get_field_descr(c2, S, 'y')
        descr_z = get_field_descr(c2, S, 'z')
        descr_f = get_field_descr(c2, S, 'f')
        descr_s = get_field_descr(c2, S, 's')
        assert descr_x.__class__ is cls
        assert descr_y.__class__ is GcPtrFieldDescr
        assert descr_z.__class__ is NonGcPtrFieldDescr
        assert descr_f.__class__ is clsf
        assert descr_s.__class__ is clss
        assert descr_x.name == 'S.x'
        assert descr_y.name == 'S.y'
        assert descr_z.name == 'S.z'
        assert descr_f.name == 'S.f'
        assert descr_s.name == 'S.s'
        if not tsc:
            assert descr_x.offset < descr_y.offset < descr_z.offset
            assert descr_x.sort_key() < descr_y.sort_key() < descr_z.sort_key()
            assert descr_x.get_field_size(False) == rffi.sizeof(lltype.Char)
            assert descr_y.get_field_size(False) == rffi.sizeof(lltype.Ptr(T))
            assert descr_z.get_field_size(False) == rffi.sizeof(lltype.Ptr(U))
            assert descr_f.get_field_size(False) == rffi.sizeof(lltype.Float)
            assert descr_s.get_field_size(False) == rffi.sizeof(
                                                            lltype.SingleFloat)
        else:
            assert isinstance(descr_x.offset, Symbolic)
            assert isinstance(descr_y.offset, Symbolic)
            assert isinstance(descr_z.offset, Symbolic)
            assert isinstance(descr_f.offset, Symbolic)
            assert isinstance(descr_s.offset, Symbolic)
            assert isinstance(descr_x.get_field_size(True), Symbolic)
            assert isinstance(descr_y.get_field_size(True), Symbolic)
            assert isinstance(descr_z.get_field_size(True), Symbolic)
            assert isinstance(descr_f.get_field_size(True), Symbolic)
            assert isinstance(descr_s.get_field_size(True), Symbolic)
        assert not descr_x.is_pointer_field()
        assert     descr_y.is_pointer_field()
        assert not descr_z.is_pointer_field()
        assert not descr_f.is_pointer_field()
        assert not descr_s.is_pointer_field()
        assert not descr_x.is_float_field()
        assert not descr_y.is_float_field()
        assert not descr_z.is_float_field()
        assert     descr_f.is_float_field()
        assert not descr_s.is_float_field()


def test_get_field_descr_sign():
    for RESTYPE, signed in [(rffi.SIGNEDCHAR, True), (rffi.UCHAR,  False),
                            (rffi.SHORT,      True), (rffi.USHORT, False),
                            (rffi.INT,        True), (rffi.UINT,   False),
                            (rffi.LONG,       True), (rffi.ULONG,  False)]:
        S = lltype.GcStruct('S', ('x', RESTYPE))
        for tsc in [False, True]:
            c2 = GcCache(tsc)
            descr_x = get_field_descr(c2, S, 'x')
            assert descr_x.is_field_signed() == signed

def test_get_field_descr_longlong():
    if sys.maxint > 2147483647:
        py.test.skip("long long: for 32-bit only")
    c0 = GcCache(False)
    S = lltype.GcStruct('S', ('y', lltype.UnsignedLongLong))
    descr = get_field_descr(c0, S, 'y')
    assert not descr.is_pointer_field()
    assert descr.is_float_field()
    assert descr.get_field_size(False) == 8


def test_get_array_descr():
    U = lltype.Struct('U')
    T = lltype.GcStruct('T')
    A1 = lltype.GcArray(lltype.Char)
    A2 = lltype.GcArray(lltype.Ptr(T))
    A3 = lltype.GcArray(lltype.Ptr(U))
    A4 = lltype.GcArray(lltype.Float)
    A5 = lltype.GcArray(lltype.Struct('x', ('v', lltype.Signed),
                                      ('k', lltype.Signed)))
    A6 = lltype.GcArray(lltype.SingleFloat)
    assert getArrayDescrClass(A2) is GcPtrArrayDescr
    assert getArrayDescrClass(A3) is NonGcPtrArrayDescr
    cls = getArrayDescrClass(A1)
    assert cls != getArrayDescrClass(lltype.GcArray(lltype.Signed))
    assert cls == getArrayDescrClass(lltype.GcArray(lltype.Char))
    clsf = getArrayDescrClass(A4)
    assert clsf != cls
    assert clsf == getArrayDescrClass(lltype.GcArray(lltype.Float))
    clss = getArrayDescrClass(A6)
    assert clss not in (clsf, cls)
    assert clss == getArrayDescrClass(lltype.GcArray(rffi.UINT))
    #
    c0 = GcCache(False)
    descr1 = get_array_descr(c0, A1)
    descr2 = get_array_descr(c0, A2)
    descr3 = get_array_descr(c0, A3)
    descr4 = get_array_descr(c0, A4)
    descr5 = get_array_descr(c0, A5)
    descr6 = get_array_descr(c0, A6)
    assert descr1.__class__ is cls
    assert descr2.__class__ is GcPtrArrayDescr
    assert descr3.__class__ is NonGcPtrArrayDescr
    assert descr4.__class__ is clsf
    assert descr6.__class__ is clss
    assert descr1 == get_array_descr(c0, lltype.GcArray(lltype.Char))
    assert not descr1.is_array_of_pointers()
    assert     descr2.is_array_of_pointers()
    assert not descr3.is_array_of_pointers()
    assert not descr4.is_array_of_pointers()
    assert not descr5.is_array_of_pointers()
    assert not descr1.is_array_of_floats()
    assert not descr2.is_array_of_floats()
    assert not descr3.is_array_of_floats()
    assert     descr4.is_array_of_floats()
    assert not descr5.is_array_of_floats()
    #
    def get_alignment(code):
        # Retrieve default alignment for the compiler/platform
        return struct.calcsize('l' + code) - struct.calcsize(code)
    assert descr1.get_base_size(False) == get_alignment('c')
    assert descr2.get_base_size(False) == get_alignment('p')
    assert descr3.get_base_size(False) == get_alignment('p')
    assert descr4.get_base_size(False) == get_alignment('d')
    assert descr5.get_base_size(False) == get_alignment('f')
    assert descr1.get_ofs_length(False) == 0
    assert descr2.get_ofs_length(False) == 0
    assert descr3.get_ofs_length(False) == 0
    assert descr4.get_ofs_length(False) == 0
    assert descr5.get_ofs_length(False) == 0
    assert descr1.get_item_size(False) == rffi.sizeof(lltype.Char)
    assert descr2.get_item_size(False) == rffi.sizeof(lltype.Ptr(T))
    assert descr3.get_item_size(False) == rffi.sizeof(lltype.Ptr(U))
    assert descr4.get_item_size(False) == rffi.sizeof(lltype.Float)
    assert descr5.get_item_size(False) == rffi.sizeof(lltype.Signed) * 2
    assert descr6.get_item_size(False) == rffi.sizeof(lltype.SingleFloat)
    #
    assert isinstance(descr1.get_base_size(True), Symbolic)
    assert isinstance(descr2.get_base_size(True), Symbolic)
    assert isinstance(descr3.get_base_size(True), Symbolic)
    assert isinstance(descr4.get_base_size(True), Symbolic)
    assert isinstance(descr5.get_base_size(True), Symbolic)
    assert isinstance(descr1.get_ofs_length(True), Symbolic)
    assert isinstance(descr2.get_ofs_length(True), Symbolic)
    assert isinstance(descr3.get_ofs_length(True), Symbolic)
    assert isinstance(descr4.get_ofs_length(True), Symbolic)
    assert isinstance(descr5.get_ofs_length(True), Symbolic)
    assert isinstance(descr1.get_item_size(True), Symbolic)
    assert isinstance(descr2.get_item_size(True), Symbolic)
    assert isinstance(descr3.get_item_size(True), Symbolic)
    assert isinstance(descr4.get_item_size(True), Symbolic)
    assert isinstance(descr5.get_item_size(True), Symbolic)
    CA = rffi.CArray(lltype.Signed)
    descr = get_array_descr(c0, CA)
    assert not descr.is_array_of_floats()
    assert descr.get_base_size(False) == 0
    assert descr.get_ofs_length(False) == -1
    CA = rffi.CArray(lltype.Ptr(lltype.GcStruct('S')))
    descr = get_array_descr(c0, CA)
    assert descr.is_array_of_pointers()
    assert descr.get_base_size(False) == 0
    assert descr.get_ofs_length(False) == -1
    CA = rffi.CArray(lltype.Ptr(lltype.Struct('S')))
    descr = get_array_descr(c0, CA)
    assert descr.get_base_size(False) == 0
    assert descr.get_ofs_length(False) == -1
    CA = rffi.CArray(lltype.Float)
    descr = get_array_descr(c0, CA)
    assert descr.is_array_of_floats()
    assert descr.get_base_size(False) == 0
    assert descr.get_ofs_length(False) == -1
    CA = rffi.CArray(rffi.FLOAT)
    descr = get_array_descr(c0, CA)
    assert not descr.is_array_of_floats()
    assert descr.get_base_size(False) == 0
    assert descr.get_ofs_length(False) == -1


def test_get_array_descr_sign():
    for RESTYPE, signed in [(rffi.SIGNEDCHAR, True), (rffi.UCHAR,  False),
                            (rffi.SHORT,      True), (rffi.USHORT, False),
                            (rffi.INT,        True), (rffi.UINT,   False),
                            (rffi.LONG,       True), (rffi.ULONG,  False)]:
        A = lltype.GcArray(RESTYPE)
        for tsc in [False, True]:
            c2 = GcCache(tsc)
            arraydescr = get_array_descr(c2, A)
            assert arraydescr.is_item_signed() == signed
        #
        RA = rffi.CArray(RESTYPE)
        for tsc in [False, True]:
            c2 = GcCache(tsc)
            arraydescr = get_array_descr(c2, RA)
            assert arraydescr.is_item_signed() == signed


def test_get_call_descr_not_translated():
    c0 = GcCache(False)
    descr1 = get_call_descr(c0, [lltype.Char, lltype.Signed], lltype.Char)
    assert descr1.get_result_size(False) == rffi.sizeof(lltype.Char)
    assert descr1.get_return_type() == history.INT
    assert descr1.arg_classes == "ii"
    #
    T = lltype.GcStruct('T')
    descr2 = get_call_descr(c0, [lltype.Ptr(T)], lltype.Ptr(T))
    assert descr2.get_result_size(False) == rffi.sizeof(lltype.Ptr(T))
    assert descr2.get_return_type() == history.REF
    assert descr2.arg_classes == "r"
    #
    U = lltype.GcStruct('U', ('x', lltype.Signed))
    assert descr2 == get_call_descr(c0, [lltype.Ptr(U)], lltype.Ptr(U))
    #
    V = lltype.Struct('V', ('x', lltype.Signed))
    assert (get_call_descr(c0, [], lltype.Ptr(V)).get_return_type() ==
            history.INT)
    #
    assert (get_call_descr(c0, [], lltype.Void).get_return_type() ==
            history.VOID)
    #
    descr4 = get_call_descr(c0, [lltype.Float, lltype.Float], lltype.Float)
    assert descr4.get_result_size(False) == rffi.sizeof(lltype.Float)
    assert descr4.get_return_type() == history.FLOAT
    assert descr4.arg_classes == "ff"
    #
    descr5 = get_call_descr(c0, [lltype.SingleFloat], lltype.SingleFloat)
    assert descr5.get_result_size(False) == rffi.sizeof(lltype.SingleFloat)
    assert descr5.get_return_type() == "S"
    assert descr5.arg_classes == "S"

def test_get_call_descr_not_translated_longlong():
    if sys.maxint > 2147483647:
        py.test.skip("long long: for 32-bit only")
    c0 = GcCache(False)
    #
    descr5 = get_call_descr(c0, [lltype.SignedLongLong], lltype.Signed)
    assert descr5.get_result_size(False) == 4
    assert descr5.get_return_type() == history.INT
    assert descr5.arg_classes == "L"
    #
    descr6 = get_call_descr(c0, [lltype.Signed], lltype.SignedLongLong)
    assert descr6.get_result_size(False) == 8
    assert descr6.get_return_type() == "L"
    assert descr6.arg_classes == "i"

def test_get_call_descr_translated():
    c1 = GcCache(True)
    T = lltype.GcStruct('T')
    U = lltype.GcStruct('U', ('x', lltype.Signed))
    descr3 = get_call_descr(c1, [lltype.Ptr(T)], lltype.Ptr(U))
    assert isinstance(descr3.get_result_size(True), Symbolic)
    assert descr3.get_return_type() == history.REF
    assert descr3.arg_classes == "r"
    #
    descr4 = get_call_descr(c1, [lltype.Float, lltype.Float], lltype.Float)
    assert isinstance(descr4.get_result_size(True), Symbolic)
    assert descr4.get_return_type() == history.FLOAT
    assert descr4.arg_classes == "ff"
    #
    descr5 = get_call_descr(c1, [lltype.SingleFloat], lltype.SingleFloat)
    assert isinstance(descr5.get_result_size(True), Symbolic)
    assert descr5.get_return_type() == "S"
    assert descr5.arg_classes == "S"

def test_call_descr_extra_info():
    c1 = GcCache(True)
    T = lltype.GcStruct('T')
    U = lltype.GcStruct('U', ('x', lltype.Signed))
    descr1 = get_call_descr(c1, [lltype.Ptr(T)], lltype.Ptr(U), "hello")
    extrainfo = descr1.get_extra_info()
    assert extrainfo == "hello"
    descr2 = get_call_descr(c1, [lltype.Ptr(T)], lltype.Ptr(U), "hello")
    assert descr1 is descr2
    descr3 = get_call_descr(c1, [lltype.Ptr(T)], lltype.Ptr(U))
    extrainfo = descr3.get_extra_info()
    assert extrainfo is None

def test_get_call_descr_sign():
    for RESTYPE, signed in [(rffi.SIGNEDCHAR, True), (rffi.UCHAR,  False),
                            (rffi.SHORT,      True), (rffi.USHORT, False),
                            (rffi.INT,        True), (rffi.UINT,   False),
                            (rffi.LONG,       True), (rffi.ULONG,  False)]:
        for tsc in [False, True]:
            c2 = GcCache(tsc)
            descr1 = get_call_descr(c2, [], RESTYPE)
            assert descr1.is_result_signed() == signed


def test_repr_of_descr():
    c0 = GcCache(False)
    T = lltype.GcStruct('T')
    S = lltype.GcStruct('S', ('x', lltype.Char),
                             ('y', lltype.Ptr(T)),
                             ('z', lltype.Ptr(T)))
    descr1 = get_size_descr(c0, S)
    s = symbolic.get_size(S, False)
    assert descr1.repr_of_descr() == '<SizeDescr %d>' % s
    #
    descr2 = get_field_descr(c0, S, 'y')
    o, _ = symbolic.get_field_token(S, 'y', False)
    assert descr2.repr_of_descr() == '<GcPtrFieldDescr S.y %d>' % o
    #
    descr2i = get_field_descr(c0, S, 'x')
    o, _ = symbolic.get_field_token(S, 'x', False)
    assert descr2i.repr_of_descr() == '<CharFieldDescr S.x %d>' % o
    #
    descr3 = get_array_descr(c0, lltype.GcArray(lltype.Ptr(S)))
    assert descr3.repr_of_descr() == '<GcPtrArrayDescr>'
    #
    descr3i = get_array_descr(c0, lltype.GcArray(lltype.Char))
    assert descr3i.repr_of_descr() == '<CharArrayDescr>'
    #
    descr4 = get_call_descr(c0, [lltype.Char, lltype.Ptr(S)], lltype.Ptr(S))
    assert 'GcPtrCallDescr' in descr4.repr_of_descr()
    #
    descr4i = get_call_descr(c0, [lltype.Char, lltype.Ptr(S)], lltype.Char)
    assert 'CharCallDescr' in descr4i.repr_of_descr()
    #
    descr4f = get_call_descr(c0, [lltype.Char, lltype.Ptr(S)], lltype.Float)
    assert 'FloatCallDescr' in descr4f.repr_of_descr()
    #
    descr5f = get_call_descr(c0, [lltype.Char], lltype.SingleFloat)
    assert 'SingleFloatCallDescr' in descr5f.repr_of_descr()

def test_call_stubs_1():
    c0 = GcCache(False)
    ARGS = [lltype.Char, lltype.Signed]
    RES = lltype.Char
    descr1 = get_call_descr(c0, ARGS, RES)
    def f(a, b):
        return 'c'

    call_stub = descr1.call_stub
    fnptr = llhelper(lltype.Ptr(lltype.FuncType(ARGS, RES)), f)

    res = call_stub(rffi.cast(lltype.Signed, fnptr), [1, 2], None, None)
    assert res == ord('c')

def test_call_stubs_2():
    c0 = GcCache(False)
    ARRAY = lltype.GcArray(lltype.Signed)
    ARGS = [lltype.Float, lltype.Ptr(ARRAY)]
    RES = lltype.Float

    def f2(a, b):
        return float(b[0]) + a

    fnptr = llhelper(lltype.Ptr(lltype.FuncType(ARGS, RES)), f2)
    descr2 = get_call_descr(c0, ARGS, RES)
    a = lltype.malloc(ARRAY, 3)
    opaquea = lltype.cast_opaque_ptr(llmemory.GCREF, a)
    a[0] = 1
    res = descr2.call_stub(rffi.cast(lltype.Signed, fnptr),
                           [], [opaquea], [longlong.getfloatstorage(3.5)])
    assert longlong.getrealfloat(res) == 4.5

def test_call_stubs_single_float():
    from pypy.rlib.longlong2float import uint2singlefloat, singlefloat2uint
    from pypy.rlib.rarithmetic import r_singlefloat, intmask
    #
    c0 = GcCache(False)
    ARGS = [lltype.SingleFloat, lltype.SingleFloat, lltype.SingleFloat]
    RES = lltype.SingleFloat

    def f(a, b, c):
        a = float(a)
        b = float(b)
        c = float(c)
        x = a - (b / c)
        return r_singlefloat(x)

    fnptr = llhelper(lltype.Ptr(lltype.FuncType(ARGS, RES)), f)
    descr2 = get_call_descr(c0, ARGS, RES)
    a = intmask(singlefloat2uint(r_singlefloat(-10.0)))
    b = intmask(singlefloat2uint(r_singlefloat(3.0)))
    c = intmask(singlefloat2uint(r_singlefloat(2.0)))
    res = descr2.call_stub(rffi.cast(lltype.Signed, fnptr),
                           [a, b, c], [], [])
    assert float(uint2singlefloat(rffi.r_uint(res))) == -11.5
