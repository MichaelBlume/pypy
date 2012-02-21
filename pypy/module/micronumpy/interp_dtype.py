
import sys
from pypy.interpreter.baseobjspace import Wrappable
from pypy.interpreter.error import OperationError
from pypy.interpreter.gateway import interp2app, unwrap_spec
from pypy.interpreter.typedef import (TypeDef, GetSetProperty,
    interp_attrproperty, interp_attrproperty_w)
from pypy.module.micronumpy import types, interp_boxes
from pypy.rlib.objectmodel import specialize
from pypy.rlib.rarithmetic import LONG_BIT, r_longlong, r_ulonglong


UNSIGNEDLTR = "u"
SIGNEDLTR = "i"
BOOLLTR = "b"
FLOATINGLTR = "f"
VOIDLTR = 'V'
STRINGLTR = 'S'
UNICODELTR = 'U'

class W_Dtype(Wrappable):
    _immutable_fields_ = ["itemtype", "num", "kind"]

    def __init__(self, itemtype, num, kind, name, char, w_box_type, alternate_constructors=[], aliases=[],
                 fields=None, fieldnames=None):
        self.itemtype = itemtype
        self.num = num
        self.kind = kind
        self.name = name
        self.char = char
        self.w_box_type = w_box_type
        self.alternate_constructors = alternate_constructors
        self.aliases = aliases
        self.fields = fields
        self.fieldnames = fieldnames

    @specialize.argtype(1)
    def box(self, value):
        return self.itemtype.box(value)

    def coerce(self, space, w_item):
        return self.itemtype.coerce(space, self, w_item)

    def getitem(self, arr, i):
        return self.itemtype.read(arr, 1, i, 0)

    def getitem_bool(self, arr, i):
        return self.itemtype.read_bool(arr, 1, i, 0)

    def setitem(self, arr, i, box):
        self.itemtype.store(arr, 1, i, 0, box)

    def fill(self, storage, box, start, stop):
        self.itemtype.fill(storage, self.get_size(), box, start, stop, 0)

    def descr_str(self, space):
        return space.wrap(self.name)

    def descr_repr(self, space):
        return space.wrap("dtype('%s')" % self.name)

    def descr_get_itemsize(self, space):
        return space.wrap(self.itemtype.get_element_size())

    def descr_get_alignment(self, space):
        return space.wrap(self.itemtype.alignment)

    def descr_get_shape(self, space):
        return space.newtuple([])

    def eq(self, space, w_other):
        w_other = space.call_function(space.gettypefor(W_Dtype), w_other)
        return space.is_w(self, w_other)

    def descr_eq(self, space, w_other):
        return space.wrap(self.eq(space, w_other))

    def descr_ne(self, space, w_other):
        return space.wrap(not self.eq(space, w_other))

    def descr_get_fields(self, space):
        if self.fields is None:
            return space.w_None
        w_d = space.newdict()
        for name, (offset, subdtype) in self.fields.iteritems():
            space.setitem(w_d, space.wrap(name), space.newtuple([subdtype,
                                                                 space.wrap(offset)]))
        return w_d

    def descr_get_names(self, space):
        if self.fieldnames is None:
            return space.w_None
        return space.newtuple([space.wrap(name) for name in self.fieldnames])

    @unwrap_spec(item=str)
    def descr_getitem(self, space, item):
        if self.fields is None:
            raise OperationError(space.w_KeyError, space.wrap("There are no keys in dtypes %s" % self.name))
        try:
            return self.fields[item][1]
        except KeyError:
            raise OperationError(space.w_KeyError, space.wrap("Field named %s not found" % item))

    def is_int_type(self):
        return (self.kind == SIGNEDLTR or self.kind == UNSIGNEDLTR or
                self.kind == BOOLLTR)

    def is_signed(self):
        return self.kind == SIGNEDLTR

    def is_bool_type(self):
        return self.kind == BOOLLTR

    def is_record_type(self):
        return self.fields is not None

    def __repr__(self):
        if self.fields is not None:
            return '<DType %r>' % self.fields
        return '<DType %r>' % self.itemtype

    def get_size(self):
        return self.itemtype.get_element_size()

def dtype_from_list(space, w_lst):
    lst_w = space.listview(w_lst)
    fields = {}
    offset = 0
    ofs_and_items = []
    fieldnames = []
    for w_elem in lst_w:
        w_fldname, w_flddesc = space.fixedview(w_elem, 2)
        subdtype = descr__new__(space, space.gettypefor(W_Dtype), w_flddesc)
        fldname = space.str_w(w_fldname)
        if fldname in fields:
            raise OperationError(space.w_ValueError, space.wrap("two fields with the same name"))
        assert isinstance(subdtype, W_Dtype)
        fields[fldname] = (offset, subdtype)
        ofs_and_items.append((offset, subdtype.itemtype))
        offset += subdtype.itemtype.get_element_size()
        fieldnames.append(fldname)
    itemtype = types.RecordType(ofs_and_items, offset)
    return W_Dtype(itemtype, 20, VOIDLTR, "void" + str(8 * itemtype.get_element_size()),
                   "V", space.gettypefor(interp_boxes.W_VoidBox), fields=fields,
                   fieldnames=fieldnames)

def dtype_from_dict(space, w_dict):
    raise OperationError(space.w_NotImplementedError, space.wrap(
        "dtype from dict"))

def variable_dtype(space, name):
    if name[0] in '<>':
        # ignore byte order, not sure if it's worth it for unicode only
        if name[0] != byteorder_prefix and name[1] == 'U':
            raise OperationError(space.w_NotImplementedError, space.wrap(
                "unimplemented non-native unicode"))
        name = name[1:]
    char = name[0]
    if len(name) == 1:
        size = 0
    else:
        try:
            size = int(name[1:])
        except ValueError:
            raise OperationError(space.w_TypeError, space.wrap("data type not understood"))
    if char == 'S':
        itemtype = types.StringType(size)
        basename = 'string'
        num = 18
        w_box_type = space.gettypefor(interp_boxes.W_StringBox)
    elif char == 'V':
        num = 20
        basename = 'void'
        w_box_type = space.gettypefor(interp_boxes.W_VoidBox)
        raise OperationError(space.w_NotImplementedError, space.wrap(
            "pure void dtype"))
    else:
        assert char == 'U'
        basename = 'unicode'
        itemtype = types.UnicodeType(size)
        num = 19
        w_box_type = space.gettypefor(interp_boxes.W_UnicodeBox)
    return W_Dtype(itemtype, num, char,
                   basename + str(8 * itemtype.get_element_size()),
                   char, w_box_type)

def dtype_from_spec(space, name):
        raise OperationError(space.w_NotImplementedError, space.wrap(
            "dtype from spec"))    

def descr__new__(space, w_subtype, w_dtype):
    cache = get_dtype_cache(space)

    if space.is_w(w_dtype, space.w_None):
        return cache.w_float64dtype
    elif space.isinstance_w(w_dtype, w_subtype):
        return w_dtype
    elif space.isinstance_w(w_dtype, space.w_str):
        name = space.str_w(w_dtype)
        if ',' in name:
            return dtype_from_spec(space, name)
        try:
            return cache.dtypes_by_name[name]
        except KeyError:
            pass
        if name[0] in 'VSU' or name[0] in '<>' and name[1] in 'VSU':
            return variable_dtype(space, name)
    elif space.isinstance_w(w_dtype, space.w_list):
        return dtype_from_list(space, w_dtype)
    elif space.isinstance_w(w_dtype, space.w_dict):
        return dtype_from_dict(space, w_dtype)
    else:
        for dtype in cache.builtin_dtypes:
            if w_dtype in dtype.alternate_constructors:
                return dtype
            if w_dtype is dtype.w_box_type:
                return dtype
    raise OperationError(space.w_TypeError, space.wrap("data type not understood"))

W_Dtype.typedef = TypeDef("dtype",
    __module__ = "numpypy",
    __new__ = interp2app(descr__new__),

    __str__= interp2app(W_Dtype.descr_str),
    __repr__ = interp2app(W_Dtype.descr_repr),
    __eq__ = interp2app(W_Dtype.descr_eq),
    __ne__ = interp2app(W_Dtype.descr_ne),
    __getitem__ = interp2app(W_Dtype.descr_getitem),

    num = interp_attrproperty("num", cls=W_Dtype),
    kind = interp_attrproperty("kind", cls=W_Dtype),
    char = interp_attrproperty("char", cls=W_Dtype),
    type = interp_attrproperty_w("w_box_type", cls=W_Dtype),
    itemsize = GetSetProperty(W_Dtype.descr_get_itemsize),
    alignment = GetSetProperty(W_Dtype.descr_get_alignment),
    shape = GetSetProperty(W_Dtype.descr_get_shape),
    name = interp_attrproperty('name', cls=W_Dtype),
    fields = GetSetProperty(W_Dtype.descr_get_fields),
    names = GetSetProperty(W_Dtype.descr_get_names),
)
W_Dtype.typedef.acceptable_as_base_class = False

if sys.byteorder == 'little':
    byteorder_prefix = '<'
    nonnative_byteorder_prefix = '>'
else:
    byteorder_prefix = '>'
    nonnative_byteorder_prefix = '<'

class DtypeCache(object):
    def __init__(self, space):
        self.w_booldtype = W_Dtype(
            types.Bool(),
            num=0,
            kind=BOOLLTR,
            name="bool",
            char="?",
            w_box_type=space.gettypefor(interp_boxes.W_BoolBox),
            alternate_constructors=[space.w_bool],
        )
        self.w_int8dtype = W_Dtype(
            types.Int8(),
            num=1,
            kind=SIGNEDLTR,
            name="int8",
            char="b",
            w_box_type=space.gettypefor(interp_boxes.W_Int8Box)
        )
        self.w_uint8dtype = W_Dtype(
            types.UInt8(),
            num=2,
            kind=UNSIGNEDLTR,
            name="uint8",
            char="B",
            w_box_type=space.gettypefor(interp_boxes.W_UInt8Box),
        )
        self.w_int16dtype = W_Dtype(
            types.Int16(),
            num=3,
            kind=SIGNEDLTR,
            name="int16",
            char="h",
            w_box_type=space.gettypefor(interp_boxes.W_Int16Box),
        )
        self.w_uint16dtype = W_Dtype(
            types.UInt16(),
            num=4,
            kind=UNSIGNEDLTR,
            name="uint16",
            char="H",
            w_box_type=space.gettypefor(interp_boxes.W_UInt16Box),
        )
        self.w_int32dtype = W_Dtype(
            types.Int32(),
            num=5,
            kind=SIGNEDLTR,
            name="int32",
            char="i",
            w_box_type=space.gettypefor(interp_boxes.W_Int32Box),
       )
        self.w_uint32dtype = W_Dtype(
            types.UInt32(),
            num=6,
            kind=UNSIGNEDLTR,
            name="uint32",
            char="I",
            w_box_type=space.gettypefor(interp_boxes.W_UInt32Box),
        )
        if LONG_BIT == 32:
            name = "int32"
        elif LONG_BIT == 64:
            name = "int64"
        self.w_longdtype = W_Dtype(
            types.Long(),
            num=7,
            kind=SIGNEDLTR,
            name=name,
            char="l",
            w_box_type=space.gettypefor(interp_boxes.W_LongBox),
            alternate_constructors=[space.w_int],
        )
        self.w_ulongdtype = W_Dtype(
            types.ULong(),
            num=8,
            kind=UNSIGNEDLTR,
            name="u" + name,
            char="L",
            w_box_type=space.gettypefor(interp_boxes.W_ULongBox),
        )
        self.w_int64dtype = W_Dtype(
            types.Int64(),
            num=9,
            kind=SIGNEDLTR,
            name="int64",
            char="q",
            w_box_type=space.gettypefor(interp_boxes.W_Int64Box),
        )
        self.w_uint64dtype = W_Dtype(
            types.UInt64(),
            num=10,
            kind=UNSIGNEDLTR,
            name="uint64",
            char="Q",
            w_box_type=space.gettypefor(interp_boxes.W_UInt64Box),
        )
        self.w_float32dtype = W_Dtype(
            types.Float32(),
            num=11,
            kind=FLOATINGLTR,
            name="float32",
            char="f",
            w_box_type=space.gettypefor(interp_boxes.W_Float32Box),
        )
        self.w_float64dtype = W_Dtype(
            types.Float64(),
            num=12,
            kind=FLOATINGLTR,
            name="float64",
            char="d",
            w_box_type = space.gettypefor(interp_boxes.W_Float64Box),
            alternate_constructors=[space.w_float],
            aliases=["float"],
        )
        self.w_longlongdtype = W_Dtype(
            types.Int64(),
            num=9,
            kind=SIGNEDLTR,
            name='int64',
            char='q',
            w_box_type = space.gettypefor(interp_boxes.W_LongLongBox),
            alternate_constructors=[space.w_long],
        )
        self.w_ulonglongdtype = W_Dtype(
            types.UInt64(),
            num=10,
            kind=UNSIGNEDLTR,
            name='uint64',
            char='Q',
            w_box_type = space.gettypefor(interp_boxes.W_ULongLongBox),
        )
        self.w_stringdtype = W_Dtype(
            types.StringType(0),
            num=18,
            kind=STRINGLTR,
            name='string',
            char='S',
            w_box_type = space.gettypefor(interp_boxes.W_StringBox),
            alternate_constructors=[space.w_str],
        )
        self.w_unicodedtype = W_Dtype(
            types.UnicodeType(0),
            num=19,
            kind=UNICODELTR,
            name='unicode',
            char='U',
            w_box_type = space.gettypefor(interp_boxes.W_UnicodeBox),
            alternate_constructors=[space.w_unicode],
        )
        self.w_voiddtype = W_Dtype(
            types.VoidType(0),
            num=20,
            kind=VOIDLTR,
            name='void',
            char='V',
            w_box_type = space.gettypefor(interp_boxes.W_VoidBox),
            #alternate_constructors=[space.w_buffer],
            # XXX no buffer in space
        )
        self.builtin_dtypes = [
            self.w_booldtype, self.w_int8dtype, self.w_uint8dtype,
            self.w_int16dtype, self.w_uint16dtype, self.w_int32dtype,
            self.w_uint32dtype, self.w_longdtype, self.w_ulongdtype,
            self.w_longlongdtype, self.w_ulonglongdtype,
            self.w_float32dtype,
            self.w_float64dtype, self.w_stringdtype, self.w_unicodedtype,
            self.w_voiddtype,
        ]
        self.dtypes_by_num_bytes = sorted(
            (dtype.itemtype.get_element_size(), dtype)
            for dtype in self.builtin_dtypes
        )
        self.dtypes_by_name = {}
        for dtype in self.builtin_dtypes:
            self.dtypes_by_name[dtype.name] = dtype
            can_name = dtype.kind + str(dtype.itemtype.get_element_size())
            self.dtypes_by_name[can_name] = dtype
            self.dtypes_by_name[byteorder_prefix + can_name] = dtype
            new_name = nonnative_byteorder_prefix + can_name
            itemtypename = dtype.itemtype.__class__.__name__
            itemtype = getattr(types, 'NonNative' + itemtypename)()
            self.dtypes_by_name[new_name] = W_Dtype(
                itemtype,
                dtype.num, dtype.kind, new_name, dtype.char, dtype.w_box_type)
            for alias in dtype.aliases:
                self.dtypes_by_name[alias] = dtype
            self.dtypes_by_name[dtype.char] = dtype

        typeinfo_full = {
            'LONGLONG': self.w_int64dtype,
            'SHORT': self.w_int16dtype,
            'VOID': self.w_voiddtype,
            #'LONGDOUBLE':,
            'UBYTE': self.w_uint8dtype,
            'UINTP': self.w_ulongdtype,
            'ULONG': self.w_ulongdtype,
            'LONG': self.w_longdtype,
            'UNICODE': self.w_unicodedtype,
            #'OBJECT',
            'ULONGLONG': self.w_ulonglongdtype,
            'STRING': self.w_stringdtype,
            #'CDOUBLE',
            #'DATETIME',
            'UINT': self.w_uint32dtype,
            'INTP': self.w_longdtype,
            #'HALF',
            'BYTE': self.w_int8dtype,
            #'CFLOAT': ,
            #'TIMEDELTA',
            'INT': self.w_int32dtype,
            'DOUBLE': self.w_float64dtype,
            'USHORT': self.w_uint16dtype,
            'FLOAT': self.w_float32dtype,
            'BOOL': self.w_booldtype,
            #, 'CLONGDOUBLE']
        }
        typeinfo_partial = {
            'Generic': interp_boxes.W_GenericBox,
            #'Character': interp_boxes.W_CharacterBox,
            'Flexible': interp_boxes.W_FlexibleBox,
            'Inexact': interp_boxes.W_InexactBox,
            'Integer': interp_boxes.W_IntegerBox,
            'SignedInteger': interp_boxes.W_SignedIntegerBox,
            'UnsignedInteger': interp_boxes.W_UnsignedIntegerBox,
            #'ComplexFloating',
            'Number': interp_boxes.W_NumberBox,
            'Floating': interp_boxes.W_FloatingBox
        }
        w_typeinfo = space.newdict()
        for k, v in typeinfo_partial.iteritems():
            space.setitem(w_typeinfo, space.wrap(k), space.gettypefor(v))
        for k, dtype in typeinfo_full.iteritems():
            itemsize = dtype.itemtype.get_element_size()
            items_w = [space.wrap(dtype.char),
                       space.wrap(dtype.num),
                       space.wrap(itemsize * 8), # in case of changing
                       # number of bits per byte in the future
                       space.wrap(itemsize or 1)]
            if dtype.is_int_type():
                if dtype.kind == BOOLLTR:
                    w_maxobj = space.wrap(1)
                    w_minobj = space.wrap(0)
                elif dtype.is_signed():
                    w_maxobj = space.wrap(r_longlong((1 << (itemsize*8 - 1))
                                          - 1))
                    w_minobj = space.wrap(r_longlong(-1) << (itemsize*8 - 1))
                else:
                    w_maxobj = space.wrap(r_ulonglong(1 << (itemsize*8)) - 1)
                    w_minobj = space.wrap(0)
                items_w = items_w + [w_maxobj, w_minobj]
            items_w = items_w + [dtype.w_box_type]
                       
            w_tuple = space.newtuple(items_w)
            space.setitem(w_typeinfo, space.wrap(k), w_tuple)
        self.w_typeinfo = w_typeinfo

def get_dtype_cache(space):
    return space.fromcache(DtypeCache)
