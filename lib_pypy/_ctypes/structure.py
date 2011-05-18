
import _rawffi
from _ctypes.basics import _CData, _CDataMeta, keepalive_key,\
     store_reference, ensure_objects, CArgObject
import inspect

def names_and_fields(self, _fields_, superclass, anonymous_fields=None):
    # _fields_: list of (name, ctype, [optional_bitfield])
    if isinstance(_fields_, tuple):
        _fields_ = list(_fields_)
    for f in _fields_:
        tp = f[1]
        if not isinstance(tp, _CDataMeta):
            raise TypeError("Expected CData subclass, got %s" % (tp,))
        if isinstance(tp, StructOrUnionMeta):
            tp._make_final()

    all_fields = []
    for cls in reversed(inspect.getmro(superclass)):
        # The first field comes from the most base class
        all_fields.extend(getattr(cls, '_fields_', []))
    all_fields.extend(_fields_)
    names = [f[0] for f in all_fields]
    rawfields = []
    for f in all_fields:
        if len(f) > 2:
            rawfields.append((f[0], f[1]._ffishape, f[2]))
        else:
            rawfields.append((f[0], f[1]._ffishape))

    _set_shape(self, rawfields, self._is_union)

    fields = {}
    for i, field in enumerate(all_fields):
        name = field[0]
        value = field[1]
        fields[name] = Field(name,
                             self._ffistruct.fieldoffset(name),
                             self._ffistruct.fieldsize(name),
                             value, i)

    if anonymous_fields:
        resnames = []
        for i, field in enumerate(all_fields):
            name = field[0]
            value = field[1]
            startpos = self._ffistruct.fieldoffset(name)
            if name in anonymous_fields:
                for subname in value._names:
                    resnames.append(subname)
                    relpos = startpos + value._fieldtypes[subname].offset
                    subvalue = value._fieldtypes[subname].ctype
                    fields[subname] = Field(subname,
                                            relpos, subvalue._sizeofinstances(),
                                            subvalue, i)
            else:
                resnames.append(name)
        names = resnames
    self._names = names
    self._fieldtypes = fields

class Field(object):
    def __init__(self, name, offset, size, ctype, num):
        for k in ('name', 'offset', 'size', 'ctype', 'num'):
            self.__dict__[k] = locals()[k]

    def __setattr__(self, name, value):
        raise AttributeError(name)

    def __repr__(self):
        return "<Field '%s' offset=%d size=%d>" % (self.name, self.offset,
                                                   self.size)

# ________________________________________________________________

def _set_shape(tp, rawfields, is_union=False):
    tp._ffistruct = _rawffi.Structure(rawfields, is_union,
                                      getattr(tp, '_pack_', 0))
    tp._ffiargshape = tp._ffishape = (tp._ffistruct, 1)
    tp._fficompositesize = tp._ffistruct.size

def struct_getattr(self, name):
    if name not in ('_fields_', '_fieldtypes'):
        if hasattr(self, '_fieldtypes') and name in self._fieldtypes:
            return self._fieldtypes[name]
    return _CDataMeta.__getattribute__(self, name)

def struct_setattr(self, name, value):
    if name == '_fields_':
        if self.__dict__.get('_fields_', None) is not None:
            raise AttributeError("_fields_ is final")
        if self in [v for k, v in value]:
            raise AttributeError("Structure or union cannot contain itself")
        names_and_fields(
            self,
            value, self.__bases__[0],
            self.__dict__.get('_anonymous_', None))
        _CDataMeta.__setattr__(self, '_fields_', value)
        return
    _CDataMeta.__setattr__(self, name, value)

class StructOrUnionMeta(_CDataMeta):

    def __new__(self, name, cls, typedict):
        res = type.__new__(self, name, cls, typedict)
        if "_abstract_" in typedict:
            return res
        cls = cls or (object,)
        if isinstance(cls[0], StructOrUnionMeta):
            cls[0]._make_final()
        if '_fields_' in typedict:
            if not hasattr(typedict.get('_anonymous_', []), '__iter__'):
                raise TypeError("Anonymous field must be iterable")
            for item in typedict.get('_anonymous_', []):
                if item not in dict(typedict['_fields_']):
                    raise AttributeError("Anonymous field not found")
            names_and_fields(
                res,
                typedict['_fields_'], cls[0],
                typedict.get('_anonymous_', None))

        return res

    def _make_final(self):
        if self is StructOrUnion:
            return
        if '_fields_' not in self.__dict__:
            self._fields_ = []
            self._names = []
            self._fieldtypes = {}
            _set_shape(self, [], self._is_union)

    __getattr__ = struct_getattr
    __setattr__ = struct_setattr

    def from_address(self, address):
        instance = StructOrUnion.__new__(self)
        instance.__dict__['_buffer'] = self._ffistruct.fromaddress(address)
        return instance

    def _sizeofinstances(self):
        if not hasattr(self, '_ffistruct'):
            return 0
        return self._ffistruct.size

    def _alignmentofinstances(self):
        return self._ffistruct.alignment

    def from_param(self, value):
        if isinstance(value, tuple):
            try:
                value = self(*value)
            except Exception, e:
                # XXX CPython does not even respect the exception type
                raise RuntimeError("(%s) %s: %s" % (self.__name__, type(e), e))
        return _CDataMeta.from_param(self, value)

    def _CData_output(self, resarray, base=None, index=-1):
        res = StructOrUnion.__new__(self)
        ffistruct = self._ffistruct.fromaddress(resarray.buffer)
        res.__dict__['_buffer'] = ffistruct
        res.__dict__['_base'] = base
        res.__dict__['_index'] = index
        return res
    
    def _CData_retval(self, resbuffer):
        res = StructOrUnion.__new__(self)
        res.__dict__['_buffer'] = resbuffer
        res.__dict__['_base'] = None
        res.__dict__['_index'] = -1
        return res

class StructOrUnion(_CData):
    __metaclass__ = StructOrUnionMeta

    def __new__(cls, *args, **kwds):
        self = super(_CData, cls).__new__(cls, *args, **kwds)
        if '_abstract_' in cls.__dict__:
            raise TypeError("abstract class")
        if hasattr(cls, '_ffistruct'):
            self.__dict__['_buffer'] = self._ffistruct(autofree=True)
        return self

    def __init__(self, *args, **kwds):
        type(self)._make_final()
        if len(args) > len(self._names):
            raise TypeError("too many initializers")
        for name, arg in zip(self._names, args):
            if name in kwds:
                raise TypeError("duplicate value for argument %r" % (
                    name,))
            self.__setattr__(name, arg)
        for name, arg in kwds.items():
            self.__setattr__(name, arg)

    def _subarray(self, fieldtype, name):
        """Return a _rawffi array of length 1 whose address is the same as
        the address of the field 'name' of self."""
        address = self._buffer.fieldaddress(name)
        A = _rawffi.Array(fieldtype._ffishape)
        return A.fromaddress(address, 1)

    def __setattr__(self, name, value):
        try:
            field = self._fieldtypes[name]
        except KeyError:
            return _CData.__setattr__(self, name, value)
        fieldtype = field.ctype
        cobj = fieldtype.from_param(value)
        if ensure_objects(cobj) is not None:
            key = keepalive_key(field.num)
            store_reference(self, key, cobj._objects)
        arg = cobj._get_buffer_value()
        if fieldtype._fficompositesize is not None:
            from ctypes import memmove
            dest = self._buffer.fieldaddress(name)
            memmove(dest, arg, fieldtype._fficompositesize)
        else:
            self._buffer.__setattr__(name, arg)

    def __getattribute__(self, name):
        if name == '_fieldtypes':
            return _CData.__getattribute__(self, '_fieldtypes')
        try:
            field = self._fieldtypes[name]
        except KeyError:
            return _CData.__getattribute__(self, name)
        if field.size >> 16:
            # bitfield member, use direct access
            return self._buffer.__getattr__(name)
        else:
            fieldtype = field.ctype
            offset = field.num
            suba = self._subarray(fieldtype, name)
            return fieldtype._CData_output(suba, self, offset)

    def _get_buffer_for_param(self):
        return self

    def _get_buffer_value(self):
        return self._buffer.buffer


class StructureMeta(StructOrUnionMeta):
    _is_union = False

class Structure(StructOrUnion):
    __metaclass__ = StructureMeta
