import sys
from pypy.rpython.lltypesystem import lltype, rffi, llmemory
from pypy.translator.tool.cbuild import ExternalCompilationInfo
from pypy.jit.backend.x86.arch import WORD


def values_array(TP, size):
    ATP = lltype.GcArray(TP)
    
    class ValuesArray(object):
        SIZE = size

        def __init__(self):
            self.ar = lltype.malloc(ATP, size, zero=True, immortal=True)

        def get_addr_for_num(self, i):
            return rffi.cast(lltype.Signed, lltype.direct_ptradd(
                lltype.direct_arrayitems(self.ar), i))

        def setitem(self, i, v):
            self.ar[i] = v

        def getitem(self, i):
            return self.ar[i]

        def _freeze_(self):
            return True

    return ValuesArray()

# ____________________________________________________________

memcpy_fn = rffi.llexternal('memcpy', [llmemory.Address, llmemory.Address,
                                       rffi.SIZE_T], lltype.Void,
                            sandboxsafe=True, _nowrapper=True)
offstack_malloc_fn = rffi.llexternal('malloc', [rffi.SIZE_T],
                                     llmemory.Address,
                                     sandboxsafe=True, _nowrapper=True)
offstack_realloc_fn = rffi.llexternal('realloc', [llmemory.Address,
                                             rffi.SIZE_T], llmemory.Address,
                                      sandboxsafe=True, _nowrapper=True)
offstack_free_fn = rffi.llexternal('free', [llmemory.Address], lltype.Void,
                                   sandboxsafe=True, _nowrapper=True)

# ____________________________________________________________

if sys.platform == 'win32':
    ensure_sse2_floats = lambda : None
    # XXX check for SSE2 on win32 too
else:
    if WORD == 4:
        extra = ['-DPYPY_X86_CHECK_SSE2']
    else:
        extra = []
    ensure_sse2_floats = rffi.llexternal_use_eci(ExternalCompilationInfo(
        compile_extra = ['-msse2', '-mfpmath=sse',
                         '-DPYPY_CPU_HAS_STANDARD_PRECISION'] + extra,
        ))
