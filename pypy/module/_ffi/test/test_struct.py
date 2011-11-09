import sys
from pypy.conftest import gettestobjspace
from pypy.module._ffi.test.test_funcptr import BaseAppTestFFI
from pypy.module._ffi.interp_struct import compute_size_and_alignement, W_Field
from pypy.module._ffi.interp_ffitype import app_types


class TestStruct(object):

    class FakeSpace(object):
        def interp_w(self, cls, obj):
            return obj

    def compute(self, ffitypes_w):
        fields_w = [W_Field('<dummy>', w_ffitype) for
                    w_ffitype in ffitypes_w]
        return compute_size_and_alignement(self.FakeSpace(), fields_w)

    def sizeof(self, ffitypes_w):
        size, aligned, fields_w = self.compute(ffitypes_w)
        return size

    def test_compute_size(self):
        T = app_types
        byte_size = app_types.sbyte.sizeof()
        long_size = app_types.slong.sizeof()
        llong_size = app_types.slonglong.sizeof()
        llong_align = app_types.slonglong.get_alignment()
        #
        assert llong_align >= 4
        assert self.sizeof([T.sbyte, T.slong]) == 2*long_size
        assert self.sizeof([T.sbyte, T.slonglong]) == llong_align + llong_size
        assert self.sizeof([T.sbyte, T.sbyte, T.slonglong]) == llong_align + llong_size
        assert self.sizeof([T.sbyte, T.sbyte, T.sbyte, T.slonglong]) == llong_align + llong_size
        assert self.sizeof([T.sbyte, T.sbyte, T.sbyte, T.sbyte, T.slonglong]) == llong_align + llong_size
        assert self.sizeof([T.slonglong, T.sbyte]) == llong_size + llong_align
        assert self.sizeof([T.slonglong, T.sbyte, T.sbyte]) == llong_size + llong_align
        assert self.sizeof([T.slonglong, T.sbyte, T.sbyte, T.sbyte]) == llong_size + llong_align
        assert self.sizeof([T.slonglong, T.sbyte, T.sbyte, T.sbyte, T.sbyte]) == llong_size + llong_align

class AppTestStruct(BaseAppTestFFI):

    def setup_class(cls):
        BaseAppTestFFI.setup_class.im_func(cls)
        #
        def read_raw_mem(self, addr, typename, length):
            import ctypes
            addr = ctypes.cast(addr, ctypes.c_void_p)
            c_type = getattr(ctypes, typename)
            array_type = ctypes.POINTER(c_type * length)
            ptr_array = ctypes.cast(addr, array_type)
            array = ptr_array[0]
            lst = [array[i] for i in range(length)]
            return lst
        cls.w_read_raw_mem = cls.space.wrap(read_raw_mem)
        
    def test__StructDescr(self):
        from _ffi import _StructDescr, Field, types
        longsize = types.slong.sizeof()
        fields = [
            Field('x', types.slong),
            Field('y', types.slong),
            ]
        descr = _StructDescr('foo', fields)
        assert descr.ffitype.sizeof() == longsize*2
        assert descr.ffitype.name == 'struct foo'

    def test_alignment(self):
        from _ffi import _StructDescr, Field, types
        longsize = types.slong.sizeof()
        fields = [
            Field('x', types.sbyte),
            Field('y', types.slong),
            ]
        descr = _StructDescr('foo', fields)
        assert descr.ffitype.sizeof() == longsize*2
        assert fields[0].offset == 0
        assert fields[1].offset == longsize # aligned to WORD

    def test_missing_field(self):
        from _ffi import _StructDescr, Field, types
        longsize = types.slong.sizeof()
        fields = [
            Field('x', types.slong),
            Field('y', types.slong),
            ]
        descr = _StructDescr('foo', fields)
        struct = descr.allocate()
        raises(AttributeError, "struct.getfield('missing')")
        raises(AttributeError, "struct.setfield('missing', 42)")

    def test_getfield_setfield(self):
        from _ffi import _StructDescr, Field, types
        longsize = types.slong.sizeof()
        fields = [
            Field('x', types.slong),
            Field('y', types.slong),
            ]
        descr = _StructDescr('foo', fields)
        struct = descr.allocate()
        struct.setfield('x', 42)
        struct.setfield('y', 43)
        assert struct.getfield('x') == 42
        assert struct.getfield('y') == 43
        mem = self.read_raw_mem(struct.getaddr(), 'c_long', 2)
        assert mem == [42, 43]

    def test_getfield_setfield_signed_types(self):
        import sys
        from _ffi import _StructDescr, Field, types
        longsize = types.slong.sizeof()
        fields = [
            Field('sbyte', types.sbyte),
            Field('sshort', types.sshort),
            Field('sint', types.sint),
            Field('slong', types.slong),
            ]
        descr = _StructDescr('foo', fields)
        struct = descr.allocate()
        struct.setfield('sbyte', 128)
        assert struct.getfield('sbyte') == -128
        struct.setfield('sshort', 32768)
        assert struct.getfield('sshort') == -32768
        struct.setfield('sint', 43)
        assert struct.getfield('sint') == 43
        struct.setfield('slong', sys.maxint+1)
        assert struct.getfield('slong') == -sys.maxint-1
        struct.setfield('slong', sys.maxint*3)
        assert struct.getfield('slong') == sys.maxint-2

    def test_getfield_setfield_unsigned_types(self):
        import sys
        from _ffi import _StructDescr, Field, types
        longsize = types.slong.sizeof()
        fields = [
            Field('ubyte', types.ubyte),
            Field('ushort', types.ushort),
            Field('uint', types.uint),
            Field('ulong', types.ulong),
            ]
        descr = _StructDescr('foo', fields)
        struct = descr.allocate()
        struct.setfield('ubyte', -1)
        assert struct.getfield('ubyte') == 255
        struct.setfield('ushort', -1)
        assert struct.getfield('ushort') == 65535
        struct.setfield('uint', 43)
        assert struct.getfield('uint') == 43
        struct.setfield('ulong', -1)
        assert struct.getfield('ulong') == sys.maxint*2 + 1
        struct.setfield('ulong', sys.maxint*2 + 2)
        assert struct.getfield('ulong') == 0

    def test_getfield_setfield_longlong(self):
        import sys
        from _ffi import _StructDescr, Field, types
        longsize = types.slong.sizeof()
        fields = [
            Field('slonglong', types.slonglong),
            Field('ulonglong', types.ulonglong),
            ]
        descr = _StructDescr('foo', fields)
        struct = descr.allocate()
        struct.setfield('slonglong', 9223372036854775808)
        assert struct.getfield('slonglong') == -9223372036854775808
        struct.setfield('ulonglong', -1)
        assert struct.getfield('ulonglong') == 18446744073709551615        
        mem = self.read_raw_mem(struct.getaddr(), 'c_longlong', 2)
        assert mem == [-9223372036854775808, -1]

    def test_getfield_setfield_float(self):
        import sys
        from _ffi import _StructDescr, Field, types
        longsize = types.slong.sizeof()
        fields = [
            Field('x', types.double),
            ]
        descr = _StructDescr('foo', fields)
        struct = descr.allocate()
        struct.setfield('x', 123.4)
        assert struct.getfield('x') == 123.4
        mem = self.read_raw_mem(struct.getaddr(), 'c_double', 1)
        assert mem == [123.4]

    def test_getfield_setfield_singlefloat(self):
        import sys
        from _ffi import _StructDescr, Field, types
        longsize = types.slong.sizeof()
        fields = [
            Field('x', types.float),
            ]
        descr = _StructDescr('foo', fields)
        struct = descr.allocate()
        struct.setfield('x', 123.4) # this is a value which DOES loose
                                    # precision in a single float
        assert 0 < abs(struct.getfield('x') - 123.4) < 0.0001
        #
        struct.setfield('x', 123.5) # this is a value which does not loose
                                    # precision in a single float
        assert struct.getfield('x') == 123.5
        mem = self.read_raw_mem(struct.getaddr(), 'c_float', 1)
        assert mem == [123.5]

    def test_compute_shape(self):
        from _ffi import Structure, Field, types
        class Point(Structure):
            _fields_ = [
                Field('x', types.slong),
                Field('y', types.slong),
                ]

        longsize = types.slong.sizeof()
        assert isinstance(Point.x, Field)
        assert isinstance(Point.y, Field)
        assert Point.x.offset == 0
        assert Point.y.offset == longsize
        assert Point._struct_.ffitype.sizeof() == longsize*2
        assert Point._struct_.ffitype.name == 'struct Point'

