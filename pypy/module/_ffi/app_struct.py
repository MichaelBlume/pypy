import _ffi

class Field(object):

    def __init__(self, name, ffitype):
        self.name = name
        self.ffitype = ffitype
        self.offset = -1

    ## def __get__(self, obj, cls=None):
    ##     if obj is None:
    ##         return self
    ##     return getfield(obj._buffer, self.ffitype, self.offset)

    ## def __set__(self, obj, value):
    ##     setfield(obj._buffer, self.ffitype, self.offset, value)

class MetaStructure(type):

    def __new__(cls, name, bases, dic):
        cls._compute_shape(dic)
        return type.__new__(cls, name, bases, dic)

    @classmethod
    def _compute_shape(cls, dic):
        fields = dic.get('_fields_')
        if fields is None:
            return
        size = 0
        ffitypes = []
        for field in fields:
            field.offset = size # XXX: alignment!
            size += field.ffitype.sizeof()
            ffitypes.append(field.ffitype)
            dic[field.name] = field
        alignment = 0 # XXX
        struct_descr = _ffi._StructDescr(size, alignment, ffitypes)
        dic['_struct_'] = struct_descr


class Structure(object):
    __metaclass__ = MetaStructure
