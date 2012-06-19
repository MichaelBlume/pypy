import sys

from pypy.interpreter.error import OperationError

from pypy.rpython.lltypesystem import rffi, lltype
from pypy.rlib.rarithmetic import r_singlefloat
from pypy.rlib import jit, libffi, clibffi, rfloat

from pypy.module._rawffi.interp_rawffi import unpack_simple_shape
from pypy.module._rawffi.array import W_Array

from pypy.module.cppyy import helper, capi


def get_rawobject(space, w_obj):
    from pypy.module.cppyy.interp_cppyy import W_CPPInstance
    cppinstance = space.interp_w(W_CPPInstance, w_obj, can_be_None=True)
    if cppinstance:
        rawobject = cppinstance.get_rawobject()
        assert lltype.typeOf(rawobject) == capi.C_OBJECT
        return rawobject
    return capi.C_NULL_OBJECT

def set_rawobject(space, w_obj, address):
    from pypy.module.cppyy.interp_cppyy import W_CPPInstance
    cppinstance = space.interp_w(W_CPPInstance, w_obj, can_be_None=True)
    if cppinstance:
        assert lltype.typeOf(cppinstance._rawobject) == capi.C_OBJECT
        cppinstance._rawobject = rffi.cast(capi.C_OBJECT, address)

def get_rawobject_nonnull(space, w_obj):
    from pypy.module.cppyy.interp_cppyy import W_CPPInstance
    cppinstance = space.interp_w(W_CPPInstance, w_obj, can_be_None=True)
    if cppinstance:
        cppinstance._nullcheck()
        rawobject = cppinstance.get_rawobject()
        assert lltype.typeOf(rawobject) == capi.C_OBJECT
        return rawobject
    return capi.C_NULL_OBJECT


class TypeConverter(object):
    _immutable_ = True
    libffitype = lltype.nullptr(clibffi.FFI_TYPE_P.TO)
    uses_local = False

    name = ""

    def __init__(self, space, extra):
        pass

    def _get_raw_address(self, space, w_obj, offset):
        rawobject = get_rawobject_nonnull(space, w_obj)
        assert lltype.typeOf(rawobject) == capi.C_OBJECT
        if rawobject:
            fieldptr = capi.direct_ptradd(rawobject, offset)
        else:
            fieldptr = rffi.cast(capi.C_OBJECT, offset)
        return fieldptr

    def _is_abstract(self, space):
        raise OperationError(space.w_TypeError, space.wrap("no converter available"))

    def convert_argument(self, space, w_obj, address, call_local):
        self._is_abstract(space)

    def convert_argument_libffi(self, space, w_obj, argchain, call_local):
        from pypy.module.cppyy.interp_cppyy import FastCallNotPossible
        raise FastCallNotPossible

    def default_argument_libffi(self, space, argchain):
        from pypy.module.cppyy.interp_cppyy import FastCallNotPossible
        raise FastCallNotPossible

    def from_memory(self, space, w_obj, w_pycppclass, offset):
        self._is_abstract(space)

    def to_memory(self, space, w_obj, w_value, offset):
        self._is_abstract(space)

    def finalize_call(self, space, w_obj, call_local):
        pass

    def free_argument(self, space, arg, call_local):
        pass


class ArrayCache(object):
    def __init__(self, space):
        self.space = space
    def __getattr__(self, name):
        if name.startswith('array_'):
            typecode = name[len('array_'):]
            arr = self.space.interp_w(W_Array, unpack_simple_shape(self.space, self.space.wrap(typecode)))
            setattr(self, name, arr)
            return arr
        raise AttributeError(name)

    def _freeze_(self):
        return True

class ArrayTypeConverterMixin(object):
    _mixin_ = True
    _immutable_ = True

    def __init__(self, space, array_size):
        if array_size <= 0:
            self.size = sys.maxint
        else:
            self.size = array_size

    def from_memory(self, space, w_obj, w_pycppclass, offset):
        if hasattr(space, "fake"):
            raise NotImplementedError
        # read access, so no copy needed
        address_value = self._get_raw_address(space, w_obj, offset)
        address = rffi.cast(rffi.ULONG, address_value)
        cache = space.fromcache(ArrayCache)
        arr = getattr(cache, 'array_' + self.typecode)
        return arr.fromaddress(space, address, self.size)

    def to_memory(self, space, w_obj, w_value, offset):
        # copy the full array (uses byte copy for now)
        address = rffi.cast(rffi.CCHARP, self._get_raw_address(space, w_obj, offset))
        buf = space.buffer_w(w_value)
        # TODO: report if too many items given?
        for i in range(min(self.size*self.typesize, buf.getlength())):
            address[i] = buf.getitem(i)


class PtrTypeConverterMixin(object):
    _mixin_ = True
    _immutable_ = True

    def __init__(self, space, array_size):
        self.size = sys.maxint

    def from_memory(self, space, w_obj, w_pycppclass, offset):
        # read access, so no copy needed
        address_value = self._get_raw_address(space, w_obj, offset)
        address = rffi.cast(rffi.ULONGP, address_value)
        cache = space.fromcache(ArrayCache)
        arr = getattr(cache, 'array_' + self.typecode)
        return arr.fromaddress(space, address[0], self.size)

    def to_memory(self, space, w_obj, w_value, offset):
        # copy only the pointer value
        rawobject = get_rawobject_nonnull(space, w_obj)
        byteptr = rffi.cast(rffi.CCHARPP, capi.direct_ptradd(rawobject, offset))
        buf = space.buffer_w(w_value)
        try:
            byteptr[0] = buf.get_raw_address()
        except ValueError:
            raise OperationError(space.w_TypeError,
                                 space.wrap("raw buffer interface not supported"))


class NumericTypeConverterMixin(object):
    _mixin_ = True
    _immutable_ = True

    def convert_argument_libffi(self, space, w_obj, argchain, call_local):
        argchain.arg(self._unwrap_object(space, w_obj))

    def default_argument_libffi(self, space, argchain):
        argchain.arg(self.default)

    def from_memory(self, space, w_obj, w_pycppclass, offset):
        address = self._get_raw_address(space, w_obj, offset)
        rffiptr = rffi.cast(self.c_ptrtype, address)
        return space.wrap(rffiptr[0])

    def to_memory(self, space, w_obj, w_value, offset):
        address = self._get_raw_address(space, w_obj, offset)
        rffiptr = rffi.cast(self.c_ptrtype, address)
        rffiptr[0] = self._unwrap_object(space, w_value)

class ConstRefNumericTypeConverterMixin(NumericTypeConverterMixin):
    _mixin_ = True
    _immutable_ = True
    uses_local = True

    def convert_argument_libffi(self, space, w_obj, argchain, call_local):
        assert rffi.sizeof(self.c_type) <= 2*rffi.sizeof(rffi.VOIDP)  # see interp_cppyy.py
        obj = self._unwrap_object(space, w_obj)
        typed_buf = rffi.cast(self.c_ptrtype, call_local)
        typed_buf[0] = obj
        argchain.arg(call_local)

class IntTypeConverterMixin(NumericTypeConverterMixin):
    _mixin_ = True
    _immutable_ = True

    def convert_argument(self, space, w_obj, address, call_local):
        x = rffi.cast(self.c_ptrtype, address)
        x[0] = self._unwrap_object(space, w_obj)

class FloatTypeConverterMixin(NumericTypeConverterMixin):
    _mixin_ = True
    _immutable_ = True

    def convert_argument(self, space, w_obj, address, call_local):
        x = rffi.cast(self.c_ptrtype, address)
        x[0] = self._unwrap_object(space, w_obj)
        ba = rffi.cast(rffi.CCHARP, address)
        ba[capi.c_function_arg_typeoffset()] = self.typecode


class VoidConverter(TypeConverter):
    _immutable_ = True
    libffitype = libffi.types.void

    def __init__(self, space, name):
        self.name = name

    def convert_argument(self, space, w_obj, address, call_local):
        raise OperationError(space.w_TypeError,
                             space.wrap('no converter available for type "%s"' % self.name))


class BoolConverter(TypeConverter):
    _immutable_ = True
    libffitype = libffi.types.schar

    def _unwrap_object(self, space, w_obj):
        arg = space.c_int_w(w_obj)
        if arg != False and arg != True:
            raise OperationError(space.w_ValueError,
                                 space.wrap("boolean value should be bool, or integer 1 or 0"))
        return arg

    def convert_argument(self, space, w_obj, address, call_local):
        x = rffi.cast(rffi.LONGP, address)
        x[0] = self._unwrap_object(space, w_obj)

    def convert_argument_libffi(self, space, w_obj, argchain, call_local):
        argchain.arg(self._unwrap_object(space, w_obj))

    def from_memory(self, space, w_obj, w_pycppclass, offset):
        address = rffi.cast(rffi.CCHARP, self._get_raw_address(space, w_obj, offset))
        if address[0] == '\x01':
            return space.w_True
        return space.w_False

    def to_memory(self, space, w_obj, w_value, offset):
        address = rffi.cast(rffi.CCHARP, self._get_raw_address(space, w_obj, offset))
        arg = self._unwrap_object(space, w_value)
        if arg:
            address[0] = '\x01'
        else:
            address[0] = '\x00'

class CharConverter(TypeConverter):
    _immutable_ = True
    libffitype = libffi.types.schar

    def _unwrap_object(self, space, w_value):
        # allow int to pass to char and make sure that str is of length 1
        if space.isinstance_w(w_value, space.w_int):
            ival = space.c_int_w(w_value)
            if ival < 0 or 256 <= ival:
                raise OperationError(space.w_ValueError,
                                     space.wrap("char arg not in range(256)"))

            value = rffi.cast(rffi.CHAR, space.c_int_w(w_value))
        else:
            value = space.str_w(w_value)

        if len(value) != 1:  
            raise OperationError(space.w_ValueError,
                                 space.wrap("char expected, got string of size %d" % len(value)))
        return value[0] # turn it into a "char" to the annotator

    def convert_argument(self, space, w_obj, address, call_local):
        x = rffi.cast(rffi.CCHARP, address)
        x[0] = self._unwrap_object(space, w_obj)

    def convert_argument_libffi(self, space, w_obj, argchain, call_local):
        argchain.arg(self._unwrap_object(space, w_obj))

    def from_memory(self, space, w_obj, w_pycppclass, offset):
        address = rffi.cast(rffi.CCHARP, self._get_raw_address(space, w_obj, offset))
        return space.wrap(address[0])

    def to_memory(self, space, w_obj, w_value, offset):
        address = rffi.cast(rffi.CCHARP, self._get_raw_address(space, w_obj, offset))
        address[0] = self._unwrap_object(space, w_value)


class ShortConverter(IntTypeConverterMixin, TypeConverter):
    _immutable_ = True
    libffitype = libffi.types.sshort
    c_type     = rffi.SHORT
    c_ptrtype  = rffi.SHORTP

    def __init__(self, space, default):
        self.default = rffi.cast(rffi.SHORT, capi.c_strtoll(default))

    def _unwrap_object(self, space, w_obj):
        return rffi.cast(rffi.SHORT, space.int_w(w_obj))

class ConstShortRefConverter(ConstRefNumericTypeConverterMixin, ShortConverter):
    _immutable_ = True
    libffitype = libffi.types.pointer

class UnsignedShortConverter(IntTypeConverterMixin, TypeConverter):
    _immutable_ = True
    libffitype = libffi.types.sshort
    c_type     = rffi.USHORT
    c_ptrtype  = rffi.USHORTP

    def __init__(self, space, default):
        self.default = rffi.cast(self.c_type, capi.c_strtoull(default))

    def _unwrap_object(self, space, w_obj):
        return rffi.cast(self.c_type, space.int_w(w_obj))

class ConstUnsignedShortRefConverter(ConstRefNumericTypeConverterMixin, UnsignedShortConverter):
    _immutable_ = True
    libffitype = libffi.types.pointer

class IntConverter(IntTypeConverterMixin, TypeConverter):
    _immutable_ = True
    libffitype = libffi.types.sint
    c_type     = rffi.INT
    c_ptrtype  = rffi.INTP

    def __init__(self, space, default):
        self.default = rffi.cast(self.c_type, capi.c_strtoll(default))

    def _unwrap_object(self, space, w_obj):
        return rffi.cast(self.c_type, space.c_int_w(w_obj))

class ConstIntRefConverter(ConstRefNumericTypeConverterMixin, IntConverter):
    _immutable_ = True
    libffitype = libffi.types.pointer

class UnsignedIntConverter(IntTypeConverterMixin, TypeConverter):
    _immutable_ = True
    libffitype = libffi.types.uint
    c_type     = rffi.UINT
    c_ptrtype  = rffi.UINTP

    def __init__(self, space, default):
        self.default = rffi.cast(self.c_type, capi.c_strtoull(default))

    def _unwrap_object(self, space, w_obj):
        return rffi.cast(self.c_type, space.uint_w(w_obj))

class ConstUnsignedIntRefConverter(ConstRefNumericTypeConverterMixin, UnsignedIntConverter):
    _immutable_ = True
    libffitype = libffi.types.pointer

class LongConverter(IntTypeConverterMixin, TypeConverter):
    _immutable_ = True
    libffitype = libffi.types.slong
    c_type     = rffi.LONG
    c_ptrtype  = rffi.LONGP

    def __init__(self, space, default):
        self.default = rffi.cast(self.c_type, capi.c_strtoll(default))

    def _unwrap_object(self, space, w_obj):
        return space.int_w(w_obj)

class ConstLongRefConverter(ConstRefNumericTypeConverterMixin, LongConverter):
    _immutable_ = True
    libffitype = libffi.types.pointer
    typecode = 'r'

    def convert_argument(self, space, w_obj, address, call_local):
        x = rffi.cast(self.c_ptrtype, address)
        x[0] = self._unwrap_object(space, w_obj)
        ba = rffi.cast(rffi.CCHARP, address)
        ba[capi.c_function_arg_typeoffset()] = self.typecode

class LongLongConverter(IntTypeConverterMixin, TypeConverter):
    _immutable_ = True
    libffitype = libffi.types.slong
    c_type     = rffi.LONGLONG
    c_ptrtype  = rffi.LONGLONGP

    def __init__(self, space, default):
        self.default = rffi.cast(self.c_type, capi.c_strtoll(default))

    def _unwrap_object(self, space, w_obj):
        return space.r_longlong_w(w_obj)

class ConstLongLongRefConverter(ConstRefNumericTypeConverterMixin, LongLongConverter):
    _immutable_ = True
    libffitype = libffi.types.pointer
    typecode = 'r'

    def convert_argument(self, space, w_obj, address, call_local):
        x = rffi.cast(self.c_ptrtype, address)
        x[0] = self._unwrap_object(space, w_obj)
        ba = rffi.cast(rffi.CCHARP, address)
        ba[capi.c_function_arg_typeoffset()] = self.typecode

class UnsignedLongConverter(IntTypeConverterMixin, TypeConverter):
    _immutable_ = True
    libffitype = libffi.types.ulong
    c_type     = rffi.ULONG
    c_ptrtype  = rffi.ULONGP

    def __init__(self, space, default):
        self.default = rffi.cast(self.c_type, capi.c_strtoull(default))

    def _unwrap_object(self, space, w_obj):
        return space.uint_w(w_obj)

class ConstUnsignedLongRefConverter(ConstRefNumericTypeConverterMixin, UnsignedLongConverter):
    _immutable_ = True
    libffitype = libffi.types.pointer

class UnsignedLongLongConverter(IntTypeConverterMixin, TypeConverter):
    _immutable_ = True
    libffitype = libffi.types.ulong
    c_type     = rffi.ULONGLONG
    c_ptrtype  = rffi.ULONGLONGP

    def __init__(self, space, default):
        self.default = rffi.cast(self.c_type, capi.c_strtoull(default))

    def _unwrap_object(self, space, w_obj):
        return space.r_ulonglong_w(w_obj)

class ConstUnsignedLongLongRefConverter(ConstRefNumericTypeConverterMixin, UnsignedLongLongConverter):
    _immutable_ = True
    libffitype = libffi.types.pointer


class FloatConverter(FloatTypeConverterMixin, TypeConverter):
    _immutable_ = True
    libffitype = libffi.types.float
    c_type     = rffi.FLOAT
    c_ptrtype  = rffi.FLOATP
    typecode   = 'f'

    def __init__(self, space, default):
        if default:
            fval = float(rfloat.rstring_to_float(default))
        else:
            fval = float(0.)
        self.default = r_singlefloat(fval)

    def _unwrap_object(self, space, w_obj):
        return r_singlefloat(space.float_w(w_obj))

    def from_memory(self, space, w_obj, w_pycppclass, offset):
        address = self._get_raw_address(space, w_obj, offset)
        rffiptr = rffi.cast(self.c_ptrtype, address)
        return space.wrap(float(rffiptr[0]))

class ConstFloatRefConverter(FloatConverter):
    _immutable_ = True
    libffitype = libffi.types.pointer
    typecode = 'F'

    def convert_argument_libffi(self, space, w_obj, argchain, call_local):
        from pypy.module.cppyy.interp_cppyy import FastCallNotPossible
        raise FastCallNotPossible

class DoubleConverter(FloatTypeConverterMixin, TypeConverter):
    _immutable_ = True
    libffitype = libffi.types.double
    c_type     = rffi.DOUBLE
    c_ptrtype  = rffi.DOUBLEP
    typecode   = 'd'

    def __init__(self, space, default):
        if default:
            self.default = rffi.cast(self.c_type, rfloat.rstring_to_float(default))
        else:
            self.default = rffi.cast(self.c_type, 0.)

    def _unwrap_object(self, space, w_obj):
        return space.float_w(w_obj)

class ConstDoubleRefConverter(ConstRefNumericTypeConverterMixin, DoubleConverter):
    _immutable_ = True
    libffitype = libffi.types.pointer
    typecode = 'D'


class CStringConverter(TypeConverter):
    _immutable_ = True

    def convert_argument(self, space, w_obj, address, call_local):
        x = rffi.cast(rffi.LONGP, address)
        arg = space.str_w(w_obj)
        x[0] = rffi.cast(rffi.LONG, rffi.str2charp(arg))
        ba = rffi.cast(rffi.CCHARP, address)
        ba[capi.c_function_arg_typeoffset()] = 'o'

    def from_memory(self, space, w_obj, w_pycppclass, offset):
        address = self._get_raw_address(space, w_obj, offset)
        charpptr = rffi.cast(rffi.CCHARPP, address)
        return space.wrap(rffi.charp2str(charpptr[0]))

    def free_argument(self, space, arg, call_local):
        lltype.free(rffi.cast(rffi.CCHARPP, arg)[0], flavor='raw')


class VoidPtrConverter(TypeConverter):
    _immutable_ = True

    def convert_argument(self, space, w_obj, address, call_local):
        x = rffi.cast(rffi.VOIDPP, address)
        x[0] = rffi.cast(rffi.VOIDP, get_rawobject(space, w_obj))
        ba = rffi.cast(rffi.CCHARP, address)
        ba[capi.c_function_arg_typeoffset()] = 'a'

    def convert_argument_libffi(self, space, w_obj, argchain, call_local):
        argchain.arg(get_rawobject(space, w_obj))

class VoidPtrPtrConverter(TypeConverter):
    _immutable_ = True
    uses_local = True

    def convert_argument(self, space, w_obj, address, call_local):
        r = rffi.cast(rffi.VOIDPP, call_local)
        r[0] = rffi.cast(rffi.VOIDP, get_rawobject(space, w_obj))
        x = rffi.cast(rffi.VOIDPP, address)
        x[0] = rffi.cast(rffi.VOIDP, call_local)
        address = rffi.cast(capi.C_OBJECT, address)
        ba = rffi.cast(rffi.CCHARP, address)
        ba[capi.c_function_arg_typeoffset()] = 'a'

    def finalize_call(self, space, w_obj, call_local):
        r = rffi.cast(rffi.VOIDPP, call_local)
        set_rawobject(space, w_obj, r[0])

class VoidPtrRefConverter(TypeConverter):
    _immutable_ = True

    def convert_argument(self, space, w_obj, address, call_local):
        x = rffi.cast(rffi.VOIDPP, address)
        x[0] = rffi.cast(rffi.VOIDP, get_rawobject(space, w_obj))
        ba = rffi.cast(rffi.CCHARP, address)
        ba[capi.c_function_arg_typeoffset()] = 'r'


class InstancePtrConverter(TypeConverter):
    _immutable_ = True

    def __init__(self, space, cppclass):
        from pypy.module.cppyy.interp_cppyy import W_CPPClass
        assert isinstance(cppclass, W_CPPClass)
        self.cppclass = cppclass

    def _unwrap_object(self, space, w_obj):
        from pypy.module.cppyy.interp_cppyy import W_CPPInstance
        obj = space.interpclass_w(w_obj)
        if isinstance(obj, W_CPPInstance):
            if capi.c_is_subtype(obj.cppclass, self.cppclass):
                rawobject = obj.get_rawobject()
                offset = capi.c_base_offset(obj.cppclass, self.cppclass, rawobject, 1)
                obj_address = capi.direct_ptradd(rawobject, offset)
                return rffi.cast(capi.C_OBJECT, obj_address)
        raise OperationError(space.w_TypeError,
                             space.wrap("cannot pass %s as %s" %
                             (space.type(w_obj).getname(space, "?"), self.cppclass.name)))

    def convert_argument(self, space, w_obj, address, call_local):
        x = rffi.cast(rffi.VOIDPP, address)
        x[0] = rffi.cast(rffi.VOIDP, self._unwrap_object(space, w_obj))
        address = rffi.cast(capi.C_OBJECT, address)
        ba = rffi.cast(rffi.CCHARP, address)
        ba[capi.c_function_arg_typeoffset()] = 'o'

    def convert_argument_libffi(self, space, w_obj, argchain, call_local):
        argchain.arg(self._unwrap_object(space, w_obj))

    def from_memory(self, space, w_obj, w_pycppclass, offset):
        address = rffi.cast(capi.C_OBJECT, self._get_raw_address(space, w_obj, offset))
        from pypy.module.cppyy import interp_cppyy
        return interp_cppyy.wrap_cppobject_nocast(
            space, w_pycppclass, self.cppclass, address, isref=True, python_owns=False)

    def to_memory(self, space, w_obj, w_value, offset):
        address = rffi.cast(rffi.VOIDPP, self._get_raw_address(space, w_obj, offset))
        address[0] = rffi.cast(rffi.VOIDP, self._unwrap_object(space, w_value))

class InstanceConverter(InstancePtrConverter):
    _immutable_ = True

    def from_memory(self, space, w_obj, w_pycppclass, offset):
        address = rffi.cast(capi.C_OBJECT, self._get_raw_address(space, w_obj, offset))
        from pypy.module.cppyy import interp_cppyy
        return interp_cppyy.wrap_cppobject_nocast(
            space, w_pycppclass, self.cppclass, address, isref=False, python_owns=False)

    def to_memory(self, space, w_obj, w_value, offset):
        self._is_abstract(space)

class InstancePtrPtrConverter(InstancePtrConverter):
    _immutable_ = True
    uses_local = True

    def convert_argument(self, space, w_obj, address, call_local):
        r = rffi.cast(rffi.VOIDPP, call_local)
        r[0] = rffi.cast(rffi.VOIDP, self._unwrap_object(space, w_obj))
        x = rffi.cast(rffi.VOIDPP, address)
        x[0] = rffi.cast(rffi.VOIDP, call_local)
        address = rffi.cast(capi.C_OBJECT, address)
        ba = rffi.cast(rffi.CCHARP, address)
        ba[capi.c_function_arg_typeoffset()] = 'o'

    def from_memory(self, space, w_obj, w_pycppclass, offset):
        self._is_abstract(space)

    def to_memory(self, space, w_obj, w_value, offset):
        self._is_abstract(space)

    def finalize_call(self, space, w_obj, call_local):
        from pypy.module.cppyy.interp_cppyy import W_CPPInstance
        obj = space.interpclass_w(w_obj)
        assert isinstance(obj, W_CPPInstance)
        r = rffi.cast(rffi.VOIDPP, call_local)
        obj._rawobject = rffi.cast(capi.C_OBJECT, r[0])


class StdStringConverter(InstanceConverter):
    _immutable_ = True

    def __init__(self, space, extra):
        from pypy.module.cppyy import interp_cppyy
        cppclass = interp_cppyy.scope_byname(space, "std::string")
        InstanceConverter.__init__(self, space, cppclass)

    def _unwrap_object(self, space, w_obj):
        try:
           charp = rffi.str2charp(space.str_w(w_obj))
           arg = capi.c_charp2stdstring(charp)
           rffi.free_charp(charp)
           return arg
        except OperationError:
           arg = InstanceConverter._unwrap_object(self, space, w_obj)
           return capi.c_stdstring2stdstring(arg)

    def to_memory(self, space, w_obj, w_value, offset):
        try:
            address = rffi.cast(capi.C_OBJECT, self._get_raw_address(space, w_obj, offset))
            charp = rffi.str2charp(space.str_w(w_value))
            capi.c_assign2stdstring(address, charp)
            rffi.free_charp(charp)
            return
        except Exception:
            pass
        return InstanceConverter.to_memory(self, space, w_obj, w_value, offset)

    def free_argument(self, space, arg, call_local):
        capi.c_free_stdstring(rffi.cast(capi.C_OBJECT, rffi.cast(rffi.VOIDPP, arg)[0]))

class StdStringRefConverter(InstancePtrConverter):
    _immutable_ = True

    def __init__(self, space, extra):
        from pypy.module.cppyy import interp_cppyy
        cppclass = interp_cppyy.scope_byname(space, "std::string")
        InstancePtrConverter.__init__(self, space, cppclass)


class PyObjectConverter(TypeConverter):
    _immutable_ = True

    def convert_argument(self, space, w_obj, address, call_local):
        if hasattr(space, "fake"):
            raise NotImplementedError
        space.getbuiltinmodule("cpyext")
        from pypy.module.cpyext.pyobject import make_ref
        ref = make_ref(space, w_obj)
        x = rffi.cast(rffi.VOIDPP, address)
        x[0] = rffi.cast(rffi.VOIDP, ref);
        ba = rffi.cast(rffi.CCHARP, address)
        ba[capi.c_function_arg_typeoffset()] = 'a'

    def convert_argument_libffi(self, space, w_obj, argchain, call_local):
        if hasattr(space, "fake"):
            raise NotImplementedError
        space.getbuiltinmodule("cpyext")
        from pypy.module.cpyext.pyobject import make_ref
        ref = make_ref(space, w_obj)
        argchain.arg(rffi.cast(rffi.VOIDP, ref))

    def free_argument(self, space, arg, call_local):
        if hasattr(space, "fake"):
            raise NotImplementedError
        from pypy.module.cpyext.pyobject import Py_DecRef, PyObject
        Py_DecRef(space, rffi.cast(PyObject, rffi.cast(rffi.VOIDPP, arg)[0]))


_converters = {}         # builtin and custom types
_a_converters = {}       # array and ptr versions of above
def get_converter(space, name, default):
    # The matching of the name to a converter should follow:
    #   1) full, exact match
    #       1a) const-removed match
    #   2) match of decorated, unqualified type
    #   3) accept ref as pointer (for the stubs, const& can be
    #       by value, but that does not work for the ffi path)
    #   4) generalized cases (covers basically all user classes)
    #   5) void converter, which fails on use

    name = capi.c_resolve_name(name)

    #   1) full, exact match
    try:
        return _converters[name](space, default)
    except KeyError:
        pass

    #   1a) const-removed match
    try:
        return _converters[helper.remove_const(name)](space, default)
    except KeyError:
        pass

    #   2) match of decorated, unqualified type
    compound = helper.compound(name)
    clean_name = helper.clean_type(name)
    try:
        # array_index may be negative to indicate no size or no size found
        array_size = helper.array_size(name)
        return _a_converters[clean_name+compound](space, array_size)
    except KeyError:
        pass

    #   3) TODO: accept ref as pointer

    #   4) generalized cases (covers basically all user classes)
    from pypy.module.cppyy import interp_cppyy
    cppclass = interp_cppyy.scope_byname(space, clean_name)
    if cppclass:
        # type check for the benefit of the annotator
        from pypy.module.cppyy.interp_cppyy import W_CPPClass
        cppclass = space.interp_w(W_CPPClass, cppclass, can_be_None=False)
        if compound == "*" or compound == "&":
            return InstancePtrConverter(space, cppclass)
        elif compound == "**":
            return InstancePtrPtrConverter(space, cppclass)
        elif compound == "":
            return InstanceConverter(space, cppclass)
    elif capi.c_is_enum(clean_name):
        return UnsignedIntConverter(space, default)
    
    #   5) void converter, which fails on use
    #
    # return a void converter here, so that the class can be build even
    # when some types are unknown; this overload will simply fail on use
    return VoidConverter(space, name)


_converters["bool"]                     = BoolConverter
_converters["char"]                     = CharConverter
_converters["unsigned char"]            = CharConverter
_converters["short int"]                = ShortConverter
_converters["const short int&"]         = ConstShortRefConverter
_converters["short"]                    = _converters["short int"]
_converters["const short&"]             = _converters["const short int&"]
_converters["unsigned short int"]       = UnsignedShortConverter
_converters["const unsigned short int&"] = ConstUnsignedShortRefConverter
_converters["unsigned short"]           = _converters["unsigned short int"]
_converters["const unsigned short&"]    = _converters["const unsigned short int&"]
_converters["int"]                      = IntConverter
_converters["const int&"]               = ConstIntRefConverter
_converters["unsigned int"]             = UnsignedIntConverter
_converters["const unsigned int&"]      = ConstUnsignedIntRefConverter
_converters["long int"]                 = LongConverter
_converters["const long int&"]          = ConstLongRefConverter
_converters["long"]                     = _converters["long int"]
_converters["const long&"]              = _converters["const long int&"]
_converters["unsigned long int"]        = UnsignedLongConverter
_converters["const unsigned long int&"] = ConstUnsignedLongRefConverter
_converters["unsigned long"]            = _converters["unsigned long int"]
_converters["const unsigned long&"]     = _converters["const unsigned long int&"]
_converters["long long int"]            = LongLongConverter
_converters["const long long int&"]     = ConstLongLongRefConverter
_converters["long long"]                = _converters["long long int"]
_converters["const long long&"]         = _converters["const long long int&"]
_converters["unsigned long long int"]   = UnsignedLongLongConverter
_converters["const unsigned long long int&"] = ConstUnsignedLongLongRefConverter
_converters["unsigned long long"]       = _converters["unsigned long long int"]
_converters["const unsigned long long&"] = _converters["const unsigned long long int&"]
_converters["float"]                    = FloatConverter
_converters["const float&"]             = ConstFloatRefConverter
_converters["double"]                   = DoubleConverter
_converters["const double&"]            = ConstDoubleRefConverter
_converters["const char*"]              = CStringConverter
_converters["char*"]                    = CStringConverter
_converters["void*"]                    = VoidPtrConverter
_converters["void**"]                   = VoidPtrPtrConverter
_converters["void*&"]                   = VoidPtrRefConverter

# special cases (note: CINT backend requires the simple name 'string')
_converters["std::basic_string<char>"]           = StdStringConverter
_converters["string"]                            = _converters["std::basic_string<char>"]
_converters["const std::basic_string<char>&"]    = StdStringConverter     # TODO: shouldn't copy
_converters["const string&"]                     = _converters["const std::basic_string<char>&"]
_converters["std::basic_string<char>&"]          = StdStringRefConverter
_converters["string&"]                           = _converters["std::basic_string<char>&"]

_converters["PyObject*"]                         = PyObjectConverter
_converters["_object*"]                          = _converters["PyObject*"]

def _build_array_converters():
    "NOT_RPYTHON"
    array_info = (
        ('h', rffi.sizeof(rffi.SHORT),  ("short int", "short")),
        ('H', rffi.sizeof(rffi.USHORT), ("unsigned short int", "unsigned short")),
        ('i', rffi.sizeof(rffi.INT),    ("int",)),
        ('I', rffi.sizeof(rffi.UINT),   ("unsigned int", "unsigned")),
        ('l', rffi.sizeof(rffi.LONG),   ("long int", "long")),
        ('L', rffi.sizeof(rffi.ULONG),  ("unsigned long int", "unsigned long")),
        ('f', rffi.sizeof(rffi.FLOAT),  ("float",)),
        ('d', rffi.sizeof(rffi.DOUBLE), ("double",)),
    )

    for info in array_info:
        class ArrayConverter(ArrayTypeConverterMixin, TypeConverter):
            _immutable_ = True
            typecode = info[0]
            typesize = info[1]
        class PtrConverter(PtrTypeConverterMixin, TypeConverter):
            _immutable_ = True
            typecode = info[0]
            typesize = info[1]
        for name in info[2]:
            _a_converters[name+'[]'] = ArrayConverter
            _a_converters[name+'*']  = PtrConverter
_build_array_converters()
