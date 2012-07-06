"""
Enums.
"""

from pypy.interpreter.error import OperationError, operationerrfmt
from pypy.rpython.lltypesystem import rffi
from pypy.rlib.rarithmetic import intmask, r_ulonglong
from pypy.rlib.objectmodel import keepalive_until_here

from pypy.module._cffi_backend.ctypeprim import W_CTypePrimitiveSigned
from pypy.module._cffi_backend import misc


class W_CTypeEnum(W_CTypePrimitiveSigned):

    def __init__(self, space, name, enumerators, enumvalues):
        from pypy.module._cffi_backend.newtype import alignment
        name = "enum " + name
        size = rffi.sizeof(rffi.INT)
        align = alignment(rffi.INT)
        W_CTypePrimitiveSigned.__init__(self, space, size,
                                        name, len(name), align)
        self.enumerators2values = {}   # str -> int
        self.enumvalues2erators = {}   # int -> str
        for i in range(len(enumerators)):
            self.enumerators2values[enumerators[i]] = enumvalues[i]
            self.enumvalues2erators[enumvalues[i]] = enumerators[i]

    def _getfields(self):
        space = self.space
        lst = []
        for enumerator in self.enumerators2values:
            enumvalue = self.enumerators2values[enumerator]
            lst.append(space.newtuple([space.wrap(enumvalue),
                                       space.wrap(enumerator)]))
        w_lst = space.newlist(lst)
        space.call_method(w_lst, 'sort')
        return w_lst

    def str(self, cdataobj):
        w_res = self.convert_to_object(cdataobj._cdata)
        keepalive_until_here(cdataobj)
        return w_res

    def convert_to_object(self, cdata):
        value = intmask(misc.read_raw_signed_data(cdata, self.size))
        try:
            enumerator = self.enumvalues2erators[value]
        except KeyError:
            enumerator = '#%d' % (value,)
        return self.space.wrap(enumerator)

    def convert_from_object(self, cdata, w_ob):
        space = self.space
        try:
            return W_CTypePrimitiveSigned.convert_from_object(self, cdata,
                                                              w_ob)
        except OperationError, e:
            if not e.match(space, space.w_TypeError):
                raise
        if space.isinstance_w(w_ob, space.w_str):
            value = self.convert_enum_string_to_int(space.str_w(w_ob))
            value = r_ulonglong(value)
            misc.write_raw_integer_data(cdata, value, self.size)
        else:
            raise self._convert_error("str or int", w_ob)

    def cast_str(self, w_ob):
        space = self.space
        return self.convert_enum_string_to_int(space.str_w(w_ob))

    def convert_enum_string_to_int(self, s):
        space = self.space
        if s.startswith('#'):
            try:
                return int(s[1:])     # xxx is it RPython?
            except ValueError:
                raise OperationError(space.w_ValueError,
                                     space.wrap("invalid literal after '#'"))
        else:
            try:
                return self.enumerators2values[s]
            except KeyError:
                raise operationerrfmt(space.w_ValueError,
                                      "'%s' is not an enumerator for %s",
                                      s, self.name)
