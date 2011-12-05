import functools
import math

from pypy.module.micronumpy import interp_boxes
from pypy.objspace.std.floatobject import float2string
from pypy.rlib import rfloat, libffi, clibffi
from pypy.rlib.objectmodel import specialize
from pypy.rlib.rarithmetic import LONG_BIT, widen
from pypy.rpython.lltypesystem import lltype, rffi


def simple_unary_op(func):
    specialize.argtype(1)(func)
    @functools.wraps(func)
    def dispatcher(self, v):
        return self.box(
            func(
                self,
                self.for_computation(self.unbox(v))
            )
        )
    return dispatcher

def simple_binary_op(func):
    specialize.argtype(1, 2)(func)
    @functools.wraps(func)
    def dispatcher(self, v1, v2):
        return self.box(
            func(
                self,
                self.for_computation(self.unbox(v1)),
                self.for_computation(self.unbox(v2)),
            )
        )
    return dispatcher

def raw_unary_op(func):
    specialize.argtype(1)
    @functools.wraps(func)
    def dispatcher(self, v):
        return func(self,
            self.for_computation(self.unbox(v))
        )
    return dispatcher

def raw_binary_op(func):
    specialize.argtype(1, 2)(func)
    @functools.wraps(func)
    def dispatcher(self, v1, v2):
        return func(self,
            self.for_computation(self.unbox(v1)),
            self.for_computation(self.unbox(v2))
        )
    return dispatcher

class BaseType(object):
    def _unimplemented_ufunc(self, *args):
        raise NotImplementedError
    # add = sub = mul = div = mod = pow = eq = ne = lt = le = gt = ge = max = \
    #     min = copysign = pos = neg = abs = sign = reciprocal = fabs = floor = \
    #     exp = sin = cos = tan = arcsin = arccos = arctan = arcsinh = \
    #     arctanh = _unimplemented_ufunc

    def is_correct_box(self, box):
        return isinstance(box, self.BoxType)

class Primitive(object):
    _mixin_ = True
    def get_element_size(self):
        return rffi.sizeof(self.T)

    @specialize.argtype(1)
    def box(self, value):
        return self.BoxType(rffi.cast(self.T, value))

    def unbox(self, box):
        assert isinstance(box, self.BoxType)
        return box.value

    def coerce(self, space, w_item):
        if isinstance(w_item, self.BoxType):
            return w_item
        return self.coerce_subtype(space, space.gettypefor(self.BoxType), w_item)

    def coerce_subtype(self, space, w_subtype, w_item):
        # XXX: ugly
        w_obj = space.allocate_instance(self.BoxType, w_subtype)
        assert isinstance(w_obj, self.BoxType)
        w_obj.__init__(self._coerce(space, w_item).value)
        return w_obj

    def _coerce(self, space, w_item):
        raise NotImplementedError

    def read(self, storage, width, i, offset):
        return self.box(libffi.array_getitem(clibffi.cast_type_to_ffitype(self.T),
            width, storage, i, offset
        ))

    def store(self, storage, width, i, offset, box):
        value = self.unbox(box)
        libffi.array_setitem(clibffi.cast_type_to_ffitype(self.T),
            width, storage, i, offset, value
        )

    def fill(self, storage, width, box, start, stop, offset):
        value = self.unbox(box)
        for i in xrange(start, stop):
            libffi.array_setitem(clibffi.cast_type_to_ffitype(self.T),
                width, storage, i, offset, value
            )

    @simple_binary_op
    def add(self, v1, v2):
        return v1 + v2

    @simple_binary_op
    def sub(self, v1, v2):
        return v1 - v2

    @simple_binary_op
    def mul(self, v1, v2):
        return v1 * v2

    @simple_unary_op
    def pos(self, v):
        return +v

    @simple_unary_op
    def neg(self, v):
        return -v

    @simple_unary_op
    def abs(self, v):
        return abs(v)

    @raw_binary_op
    def eq(self, v1, v2):
        return v1 == v2

    @raw_binary_op
    def ne(self, v1, v2):
        return v1 != v2

    @raw_binary_op
    def lt(self, v1, v2):
        return v1 < v2

    @raw_binary_op
    def le(self, v1, v2):
        return v1 <= v2

    @raw_binary_op
    def gt(self, v1, v2):
        return v1 > v2

    @raw_binary_op
    def ge(self, v1, v2):
        return v1 >= v2

    def bool(self, v):
        return bool(self.for_computation(self.unbox(v)))

    @simple_binary_op
    def max(self, v1, v2):
        return max(v1, v2)

    @simple_binary_op
    def min(self, v1, v2):
        return min(v1, v2)

class Bool(BaseType, Primitive):
    T = lltype.Bool
    BoxType = interp_boxes.W_BoolBox

    True = BoxType(True)
    False = BoxType(False)

    @specialize.argtype(1)
    def box(self, value):
        box = Primitive.box(self, value)
        if box.value:
            return self.True
        else:
            return self.False

    def coerce_subtype(self, space, w_subtype, w_item):
        # Doesn't return subclasses so it can return the constants.
        return self._coerce(space, w_item)

    def _coerce(self, space, w_item):
        return self.box(space.is_true(w_item))

    def str_format(self, box):
        value = self.unbox(box)
        return "True" if value else "False"

    def for_computation(self, v):
        return int(v)

class Integer(Primitive):
    _mixin_ = True

    def _coerce(self, space, w_item):
        return self.box(space.int_w(space.int(w_item)))

    def str_format(self, box):
        value = self.unbox(box)
        return str(self.for_computation(value))

    def for_computation(self, v):
        return widen(v)

    @simple_binary_op
    def div(self, v1, v2):
        if v2 == 0:
            return 0
        return v1 / v2

    @simple_binary_op
    def mod(self, v1, v2):
        return v1 % v2

    @simple_binary_op
    def pow(self, v1, v2):
        res = 1
        while v2 > 0:
            if v2 & 1:
                res *= v1
            v2 >>= 1
            if v2 == 0:
                break
            v1 *= v1
        return res

    @simple_unary_op
    def sign(self, v):
        if v > 0:
            return 1
        elif v < 0:
            return -1
        else:
            assert v == 0
            return 0

class Int8(BaseType, Integer):
    T = rffi.SIGNEDCHAR
    BoxType = interp_boxes.W_Int8Box

class UInt8(BaseType, Integer):
    T = rffi.UCHAR
    BoxType = interp_boxes.W_UInt8Box

class Int16(BaseType, Integer):
    T = rffi.SHORT
    BoxType = interp_boxes.W_Int16Box

class UInt16(BaseType, Integer):
    T = rffi.USHORT
    BoxType = interp_boxes.W_UInt16Box

class Int32(BaseType, Integer):
    T = rffi.INT
    BoxType = interp_boxes.W_Int32Box

class UInt32(BaseType, Integer):
    T = rffi.UINT
    BoxType = interp_boxes.W_UInt32Box

class Long(BaseType, Integer):
    T = rffi.LONG
    BoxType = interp_boxes.W_LongBox

class ULong(BaseType, Integer):
    T = rffi.ULONG
    BoxType = interp_boxes.W_ULongBox

class Int64(BaseType, Integer):
    T = rffi.LONGLONG
    BoxType = interp_boxes.W_Int64Box

class UInt64(BaseType, Integer):
    T = rffi.ULONGLONG
    BoxType = interp_boxes.W_UInt64Box

class Float(Primitive):
    _mixin_ = True

    def _coerce(self, space, w_item):
        return self.box(space.float_w(space.float(w_item)))

    def str_format(self, box):
        value = self.unbox(box)
        return float2string(self.for_computation(value), "g", rfloat.DTSF_STR_PRECISION)

    def for_computation(self, v):
        return float(v)

    @simple_binary_op
    def div(self, v1, v2):
        try:
            return v1 / v2
        except ZeroDivisionError:
            if v1 == v2 == 0.0:
                return rfloat.NAN
            return rfloat.copysign(rfloat.INFINITY, v1 * v2)

    @simple_binary_op
    def mod(self, v1, v2):
        return math.fmod(v1, v2)

    @simple_binary_op
    def pow(self, v1, v2):
        return math.pow(v1, v2)

    @simple_binary_op
    def copysign(self, v1, v2):
        return math.copysign(v1, v2)

    @simple_unary_op
    def sign(self, v):
        if v == 0.0:
            return 0.0
        return rfloat.copysign(1.0, v)

    @simple_unary_op
    def fabs(self, v):
        return math.fabs(v)

    @simple_unary_op
    def reciprocal(self, v):
        if v == 0.0:
            return rfloat.copysign(rfloat.INFINITY, v)
        return 1.0 / v

    @simple_unary_op
    def floor(self, v):
        return math.floor(v)

    @simple_unary_op
    def exp(self, v):
        try:
            return math.exp(v)
        except OverflowError:
            return rfloat.INFINITY

    @simple_unary_op
    def sin(self, v):
        return math.sin(v)

    @simple_unary_op
    def cos(self, v):
        return math.cos(v)

    @simple_unary_op
    def tan(self, v):
        return math.tan(v)

    @simple_unary_op
    def arcsin(self, v):
        if not -1.0 <= v <= 1.0:
            return rfloat.NAN
        return math.asin(v)

    @simple_unary_op
    def arccos(self, v):
        if not -1.0 <= v <= 1.0:
            return rfloat.NAN
        return math.acos(v)

    @simple_unary_op
    def arctan(self, v):
        return math.atan(v)

    @simple_unary_op
    def arcsinh(self, v):
        return math.asinh(v)

    @simple_unary_op
    def arctanh(self, v):
        if v == 1.0 or v == -1.0:
            return math.copysign(rfloat.INFINITY, v)
        if not -1.0 < v < 1.0:
            return rfloat.NAN
        return math.atanh(v)

    @simple_unary_op
    def sqrt(self, v):
        try:
            return math.sqrt(v)
        except ValueError:
            return rfloat.NAN


class Float32(BaseType, Float):
    T = rffi.FLOAT
    BoxType = interp_boxes.W_Float32Box

class Float64(BaseType, Float):
    T = rffi.DOUBLE
    BoxType = interp_boxes.W_Float64Box


class BaseCompositeType(BaseType):
    def __init__(self, itemtypes):
        self.itemtypes = itemtypes

    def get_element_size(self):
        s = 0
        for itemtype in self.itemtypes:
            s += itemtype.get_element_size()
        return s

    def box(self, value):
        return self.BoxType(value)

    def unbox(self, box):
        assert isinstance(box, self.BoxType)
        return box.subboxes

    def store(self, storage, width, i, offset, box):
        subboxes = self.unbox(box)
        item_idx = 0
        for box in subboxes:
            self.itemtypes[item_idx].store(storage, width, i, offset, box)
            offset += self.itemtypes[item_idx].get_element_size()
            item_idx += 1

    def read(self, storage, width, i, offset):
        boxes = []
        for itemtype in self.itemtypes:
            boxes.append(itemtype.read(storage, width, i, offset))
            offset += itemtype.get_element_size()
        return self.box(boxes)

class Complex(BaseCompositeType):
    BoxType = interp_boxes.W_Complex128Box

    def __init__(self, itemtypes):
        BaseCompositeType.__init__(self, itemtypes)
        [self.real_type, self.imag_type] = self.itemtypes

    def coerce(self, space, w_item):
        if isinstance(w_item, self.BoxType):
            return w_item
        real, imag = space.unpackcomplex(w_item)
        return self.box([self.real_type.box(real), self.imag_type.box(imag)])

    def for_computation(self, (real, imag)):
        return [
            self.real_type.for_computation(self.real_type.unbox(real)),
            self.imag_type.for_computation(self.imag_type.unbox(imag)),
        ]

    @raw_binary_op
    def eq(self, (real1, imag1), (real2, imag2)):
        return real1 == real2 and imag1 == imag2

    @raw_unary_op
    def real(self, (real, imag)):
        return self.real_type.box(real)

    @raw_unary_op
    def imag(self, (real, imag)):
        return self.imag_type.box(imag)