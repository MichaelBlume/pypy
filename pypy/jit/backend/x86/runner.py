import py
from pypy.rpython.lltypesystem import lltype, llmemory, rffi
from pypy.rpython.lltypesystem.lloperation import llop
from pypy.rpython.llinterp import LLInterpreter
from pypy.rlib.objectmodel import we_are_translated
from pypy.jit.metainterp import history, compile
from pypy.jit.backend.x86.assembler import Assembler386
from pypy.jit.backend.x86.arch import FORCE_INDEX_OFS
from pypy.jit.backend.x86.profagent import ProfileAgent
from pypy.jit.backend.llsupport.llmodel import AbstractLLCPU
from pypy.jit.backend.x86 import regloc
import sys

from pypy.tool.ansi_print import ansi_log
log = py.log.Producer('jitbackend')
py.log.setconsumer('jitbackend', ansi_log)


class AbstractX86CPU(AbstractLLCPU):
    debug = True
    supports_floats = True
    supports_singlefloats = True

    BOOTSTRAP_TP = lltype.FuncType([], lltype.Signed)
    dont_keepalive_stuff = False # for tests
    with_threads = False

    def __init__(self, rtyper, stats, opts=None, translate_support_code=False,
                 gcdescr=None):
        if gcdescr is not None:
            gcdescr.force_index_ofs = FORCE_INDEX_OFS
        AbstractLLCPU.__init__(self, rtyper, stats, opts,
                               translate_support_code, gcdescr)

        profile_agent = ProfileAgent()
        if rtyper is not None:
            config = rtyper.annotator.translator.config
            if config.translation.jit_profiler == "oprofile":
                from pypy.jit.backend.x86 import oprofile
                if not oprofile.OPROFILE_AVAILABLE:
                    log.WARNING('oprofile support was explicitly enabled, but oprofile headers seem not to be available')
                profile_agent = oprofile.OProfileAgent()
            self.with_threads = config.translation.thread

        self.profile_agent = profile_agent

    def setup(self):
        if self.opts is not None:
            failargs_limit = self.opts.failargs_limit
        else:
            failargs_limit = 1000
        self.assembler = Assembler386(self, self.translate_support_code,
                                            failargs_limit)

    def get_on_leave_jitted_hook(self):
        return self.assembler.leave_jitted_hook

    def setup_once(self):
        self.profile_agent.startup()
        self.assembler.setup_once()

    def finish_once(self):
        self.assembler.finish_once()
        self.profile_agent.shutdown()

    def dump_loop_token(self, looptoken):
        """
        NOT_RPYTHON
        """
        from pypy.jit.backend.x86.tool.viewcode import machine_code_dump
        data = []
        label_list = [(offset, name) for name, offset in
                      looptoken._x86_ops_offset.iteritems()]
        label_list.sort()
        addr = looptoken._x86_rawstart
        src = rffi.cast(rffi.CCHARP, addr)
        for p in range(looptoken._x86_fullsize):
            data.append(src[p])
        data = ''.join(data)
        lines = machine_code_dump(data, addr, self.backend_name, label_list)
        print ''.join(lines)

    def compile_loop(self, inputargs, operations, looptoken, log=True, name=''):
        return self.assembler.assemble_loop(name, inputargs, operations,
                                            looptoken, log=log)

    def compile_bridge(self, faildescr, inputargs, operations,
                       original_loop_token, log=True):
        clt = original_loop_token.compiled_loop_token
        clt.compiling_a_bridge()
        return self.assembler.assemble_bridge(faildescr, inputargs, operations,
                                              original_loop_token, log=log)

    def set_future_value_int(self, index, intvalue):
        self.assembler.fail_boxes_int.setitem(index, intvalue)

    def set_future_value_float(self, index, floatvalue):
        self.assembler.fail_boxes_float.setitem(index, floatvalue)

    def set_future_value_ref(self, index, ptrvalue):
        self.assembler.fail_boxes_ptr.setitem(index, ptrvalue)

    def get_latest_value_int(self, index):
        return self.assembler.fail_boxes_int.getitem(index)

    def get_latest_value_float(self, index):
        return self.assembler.fail_boxes_float.getitem(index)

    def get_latest_value_ref(self, index):
        return self.assembler.fail_boxes_ptr.getitem(index)

    def get_latest_value_count(self):
        return self.assembler.fail_boxes_count

    def clear_latest_values(self, count):
        setitem = self.assembler.fail_boxes_ptr.setitem
        null = lltype.nullptr(llmemory.GCREF.TO)
        for index in range(count):
            setitem(index, null)

    def get_latest_force_token(self):
        # the FORCE_TOKEN operation and this helper both return 'ebp'.
        return self.assembler.fail_ebp

    def execute_token(self, executable_token):
        addr = executable_token._x86_bootstrap_code
        #llop.debug_print(lltype.Void, ">>>> Entering", addr)
        func = rffi.cast(lltype.Ptr(self.BOOTSTRAP_TP), addr)
        fail_index = self._execute_call(func)
        #llop.debug_print(lltype.Void, "<<<< Back")
        return self.get_fail_descr_from_number(fail_index)

    def _execute_call(self, func):
        # help flow objspace
        prev_interpreter = None
        if not self.translate_support_code:
            prev_interpreter = LLInterpreter.current_interpreter
            LLInterpreter.current_interpreter = self.debug_ll_interpreter
        res = 0
        try:
            res = func()
        finally:
            if not self.translate_support_code:
                LLInterpreter.current_interpreter = prev_interpreter
        return res

    def cast_ptr_to_int(x):
        adr = llmemory.cast_ptr_to_adr(x)
        return CPU386.cast_adr_to_int(adr)
    cast_ptr_to_int._annspecialcase_ = 'specialize:arglltype(0)'
    cast_ptr_to_int = staticmethod(cast_ptr_to_int)

    all_null_registers = lltype.malloc(rffi.LONGP.TO, 24,
                                       flavor='raw', zero=True,
                                       immortal=True)

    def force(self, addr_of_force_token):
        TP = rffi.CArrayPtr(lltype.Signed)
        addr_of_force_index = addr_of_force_token + FORCE_INDEX_OFS
        fail_index = rffi.cast(TP, addr_of_force_index)[0]
        assert fail_index >= 0, "already forced!"
        faildescr = self.get_fail_descr_from_number(fail_index)
        rffi.cast(TP, addr_of_force_index)[0] = ~fail_index
        frb = self.assembler._find_failure_recovery_bytecode(faildescr)
        bytecode = rffi.cast(rffi.UCHARP, frb)
        # start of "no gc operation!" block
        fail_index_2 = self.assembler.grab_frame_values(
            bytecode,
            addr_of_force_token,
            self.all_null_registers)
        self.assembler.leave_jitted_hook()
        # end of "no gc operation!" block
        assert fail_index == fail_index_2
        return faildescr

    def redirect_call_assembler(self, oldlooptoken, newlooptoken):
        self.assembler.redirect_call_assembler(oldlooptoken, newlooptoken)

    def invalidate_loop(self, looptoken):
        from pypy.jit.backend.x86 import codebuf
        
        for addr, tgt in looptoken.compiled_loop_token.invalidate_positions:
            mc = codebuf.MachineCodeBlockWrapper()
            mc.JMP_l(tgt)
            assert mc.get_relative_pos() == 5      # [JMP] [tgt 4 bytes]
            mc.copy_to_raw_memory(addr - 1)
        # positions invalidated
        looptoken.compiled_loop_token.invalidate_positions = []


class CPU386(AbstractX86CPU):
    backend_name = 'x86'
    WORD = 4
    NUM_REGS = 8
    CALLEE_SAVE_REGISTERS = [regloc.ebx, regloc.esi, regloc.edi]

    supports_longlong = True

    def __init__(self, *args, **kwargs):
        assert sys.maxint == (2**31 - 1)
        super(CPU386, self).__init__(*args, **kwargs)

class CPU386_NO_SSE2(CPU386):
    supports_floats = False
    supports_longlong = False

class CPU_X86_64(AbstractX86CPU):
    backend_name = 'x86_64'
    WORD = 8
    NUM_REGS = 16
    CALLEE_SAVE_REGISTERS = [regloc.ebx, regloc.r12, regloc.r13, regloc.r14, regloc.r15]

    def __init__(self, *args, **kwargs):
        assert sys.maxint == (2**63 - 1)
        super(CPU_X86_64, self).__init__(*args, **kwargs)

CPU = CPU386

# silence warnings
##history.LoopToken._x86_param_depth = 0
##history.LoopToken._x86_arglocs = (None, None)
##history.LoopToken._x86_frame_depth = 0
##history.LoopToken._x86_bootstrap_code = 0
##history.LoopToken._x86_direct_bootstrap_code = 0
##history.LoopToken._x86_loop_code = 0
##history.LoopToken._x86_debug_checksum = 0
##compile.AbstractFailDescr._x86_current_depths = (0, 0)
##compile.AbstractFailDescr._x86_adr_jump_offset = 0
