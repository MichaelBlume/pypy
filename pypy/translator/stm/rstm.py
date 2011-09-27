from pypy.rpython.lltypesystem import lltype, rffi
from pypy.translator.stm import _rffi_stm


def stm_getfield(structptr, fieldname):
    p = lltype.direct_fieldptr(structptr, fieldname)
    p = rffi.cast(rffi.VOIDPP, p)
    res = _rffi_stm.stm_read_word(p)
    return rffi.cast(lltype.Signed, res)