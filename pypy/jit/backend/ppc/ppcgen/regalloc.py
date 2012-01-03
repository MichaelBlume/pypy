from pypy.jit.backend.llsupport.regalloc import (RegisterManager, FrameManager,
                                                 TempBox, compute_vars_longevity)
from pypy.jit.backend.ppc.ppcgen.arch import (WORD, MY_COPY_OF_REGS)
from pypy.jit.backend.ppc.ppcgen.jump import remap_frame_layout_mixed
from pypy.jit.backend.ppc.ppcgen.locations import imm
from pypy.jit.backend.ppc.ppcgen.helper.regalloc import (_check_imm_arg, 
                                                         prepare_cmp_op,
                                                         prepare_unary_int_op,
                                                         prepare_binary_int_op,
                                                         prepare_binary_int_op_with_imm,
                                                         prepare_unary_cmp)
from pypy.jit.metainterp.history import (INT, REF, FLOAT, Const, ConstInt, 
                                         ConstPtr, Box)
from pypy.jit.metainterp.history import JitCellToken, TargetToken
from pypy.jit.metainterp.resoperation import rop
from pypy.jit.backend.ppc.ppcgen import locations
from pypy.rpython.lltypesystem import rffi, lltype, rstr
from pypy.jit.backend.llsupport import symbolic
from pypy.jit.codewriter.effectinfo import EffectInfo
import pypy.jit.backend.ppc.ppcgen.register as r
from pypy.jit.codewriter import heaptracker

class TempInt(TempBox):
    type = INT

    def __repr__(self):
        return "<TempInt at %s>" % (id(self),)

class TempPtr(TempBox):
    type = REF

    def __repr__(self):
        return "<TempPtr at %s>" % (id(self),)

class PPCRegisterManager(RegisterManager):
    all_regs              = r.MANAGED_REGS
    box_types             = None       # or a list of acceptable types
    no_lower_byte_regs    = all_regs
    save_around_call_regs = r.VOLATILES

    REGLOC_TO_COPY_AREA_OFS = {
        r.r0:   MY_COPY_OF_REGS + 0 * WORD,
        r.r2:   MY_COPY_OF_REGS + 1 * WORD,
        r.r3:   MY_COPY_OF_REGS + 2 * WORD,
        r.r4:   MY_COPY_OF_REGS + 3 * WORD,
        r.r5:   MY_COPY_OF_REGS + 4 * WORD,
        r.r6:   MY_COPY_OF_REGS + 5 * WORD,
        r.r7:   MY_COPY_OF_REGS + 6 * WORD,
        r.r8:   MY_COPY_OF_REGS + 7 * WORD,
        r.r9:   MY_COPY_OF_REGS + 8 * WORD,
        r.r10:  MY_COPY_OF_REGS + 9 * WORD,
        r.r11:  MY_COPY_OF_REGS + 10 * WORD,
        r.r12:  MY_COPY_OF_REGS + 11 * WORD,
        r.r13:  MY_COPY_OF_REGS + 12 * WORD,
        r.r14:  MY_COPY_OF_REGS + 13 * WORD,
        r.r15:  MY_COPY_OF_REGS + 14 * WORD,
        r.r16:  MY_COPY_OF_REGS + 15 * WORD,
        r.r17:  MY_COPY_OF_REGS + 16 * WORD,
        r.r18:  MY_COPY_OF_REGS + 17 * WORD,
        r.r19:  MY_COPY_OF_REGS + 18 * WORD,
        r.r20:  MY_COPY_OF_REGS + 19 * WORD,
        r.r21:  MY_COPY_OF_REGS + 20 * WORD,
        r.r22:  MY_COPY_OF_REGS + 21 * WORD,
        r.r23:  MY_COPY_OF_REGS + 22 * WORD,
        r.r24:  MY_COPY_OF_REGS + 23 * WORD,
        r.r25:  MY_COPY_OF_REGS + 24 * WORD,
        r.r26:  MY_COPY_OF_REGS + 25 * WORD,
        r.r27:  MY_COPY_OF_REGS + 26 * WORD,
        r.r28:  MY_COPY_OF_REGS + 27 * WORD,
        r.r29:  MY_COPY_OF_REGS + 28 * WORD,
        r.r30:  MY_COPY_OF_REGS + 29 * WORD,
        r.r31:  MY_COPY_OF_REGS + 30 * WORD,
    }

    def __init__(self, longevity, frame_manager=None, assembler=None):
        RegisterManager.__init__(self, longevity, frame_manager, assembler)

    def call_result_location(self, v):
        return r.r3

    def convert_to_imm(self, c):
        if isinstance(c, ConstInt):
            return locations.ImmLocation(c.value)
        else:
            assert isinstance(c, ConstPtr)
            return locations.ImmLocation(rffi.cast(lltype.Signed, c.value))

    def allocate_scratch_reg(self, type=INT, selected_reg=None, forbidden_vars=None):
        """Allocate a scratch register, possibly spilling a managed register.
        This register is freed after emitting the current operation and can not
        be spilled"""
        box = TempBox()
        reg = self.force_allocate_reg(box,
                            selected_reg=selected_reg,
                            forbidden_vars=forbidden_vars)
        return reg, box

class PPCFrameManager(FrameManager):
    def __init__(self):
        FrameManager.__init__(self)
        self.frame_depth = 0

    @staticmethod
    def frame_pos(loc, type):
        num_words = PPCFrameManager.frame_size(type)
        return locations.StackLocation(loc, num_words=num_words, type=type)

    @staticmethod
    def frame_size(type):
        if type == FLOAT:
            assert 0, "TODO"
        return 1

class Regalloc(object):
    def __init__(self, longevity, frame_manager=None, assembler=None):
        self.cpu = assembler.cpu
        self.longevity = longevity
        self.frame_manager = frame_manager
        self.assembler = assembler
        self.rm = PPCRegisterManager(longevity, frame_manager, assembler)

    def prepare_loop(self, inputargs, operations, looptoken):
        loop_consts = compute_loop_consts(inputargs, operations[-1], looptoken)
        inputlen = len(inputargs)
        nonfloatlocs = [None] * len(inputargs)
        for i in range(inputlen):
            arg = inputargs[i]
            assert not isinstance(arg, Const)
            if arg not in loop_consts and self.longevity[arg][1] > -1:
                self.try_allocate_reg(arg)
            loc = self.loc(arg)
            nonfloatlocs[i] = loc
        self.possibly_free_vars(inputargs)
        return nonfloatlocs

    def update_bindings(self, locs, frame_depth, inputargs):
        used = {}
        i = 0
        self.frame_manager.frame_depth = frame_depth
        for loc in locs:
            arg = inputargs[i]
            i += 1
            if loc.is_reg():
                self.rm.reg_bindings[arg] = loc
            elif loc.is_vfp_reg():
                assert 0, "not implemented yet"
            else:
                assert loc.is_stack()
                self.frame_manager.frame_bindings[arg] = loc
            used[loc] = None

        # XXX combine with x86 code and move to llsupport
        self.rm.free_regs = []
        for reg in self.rm.all_regs:
            if reg not in used:
                self.rm.free_regs.append(reg)
        # note: we need to make a copy of inputargs because possibly_free_vars
        # is also used on op args, which is a non-resizable list
        self.possibly_free_vars(list(inputargs))

    def possibly_free_var(self, var):
        self.rm.possibly_free_var(var)

    def possibly_free_vars(self, vars):
        for var in vars:
            self.possibly_free_var(var)

    def possibly_free_vars_for_op(self, op):
        for i in range(op.numargs()):
            var = op.getarg(i)
            if var is not None:
                self.possibly_free_var(var)

    def try_allocate_reg(self, v, selected_reg=None, need_lower_byte=False):
        return self.rm.try_allocate_reg(v, selected_reg, need_lower_byte)

    def force_allocate_reg(self, var, forbidden_vars=[], selected_reg=None, 
            need_lower_byte=False):
        return self.rm.force_allocate_reg(var, forbidden_vars, selected_reg,
                need_lower_byte)

    def allocate_scratch_reg(self, type=INT, forbidden_vars=[], selected_reg=None):
        assert type == INT # XXX extend this once floats are supported
        return self.rm.allocate_scratch_reg(type=type,
                        forbidden_vars=forbidden_vars,
                        selected_reg=selected_reg)

    def _check_invariants(self):
        self.rm._check_invariants()

    def loc(self, var):
        if var.type == FLOAT:
            assert 0, "not implemented yet"
        return self.rm.loc(var)

    def position(self):
        return self.rm.position

    def next_instruction(self):
        self.rm.next_instruction()

    def force_spill_var(self, var):
        if var.type == FLOAT:
            assert 0, "not implemented yet"
        else:
            self.rm.force_spill_var(var)

    def before_call(self, force_store=[], save_all_regs=False):
        self.rm.before_call(force_store, save_all_regs)

    def after_call(self, v):
        if v.type == FLOAT:
            assert 0, "not implemented yet"
        else:
            return self.rm.after_call(v)

    def call_result_location(self, v):
        if v.type == FLOAT:
            assert 0, "not implemented yet"
        else:
            return self.rm.call_result_location(v)

    def _ensure_value_is_boxed(self, thing, forbidden_vars=[]):
        box = None
        loc = None
        if isinstance(thing, Const):
            if isinstance(thing, ConstPtr):
                box = TempPtr()
            else:
                box = TempInt()
            loc = self.force_allocate_reg(box, forbidden_vars=forbidden_vars)
            imm = self.rm.convert_to_imm(thing)
            self.assembler.mc.load_imm(loc, imm.value)
        else:
            loc = self.make_sure_var_in_reg(thing,
                    forbidden_vars=forbidden_vars)
            box = thing
        return loc, box

    def make_sure_var_in_reg(self, var, forbidden_vars=[],
                             selected_reg=None, need_lower_byte=False):
        return self.rm.make_sure_var_in_reg(var, forbidden_vars,
                selected_reg, need_lower_byte)

    def _sync_var(self, v):
        if v.type == FLOAT:
            assert 0, "not implemented yet"
        else:
            self.rm._sync_var(v)

    # ******************************************************
    # *         P R E P A R E  O P E R A T I O N S         * 
    # ******************************************************


    def void(self, op):
        return []

    prepare_int_add = prepare_binary_int_op_with_imm()
    prepare_int_sub = prepare_binary_int_op_with_imm()
    prepare_int_floordiv = prepare_binary_int_op_with_imm()

    prepare_int_mul = prepare_binary_int_op()
    prepare_int_mod = prepare_binary_int_op()
    prepare_int_and = prepare_binary_int_op()
    prepare_int_or = prepare_binary_int_op()
    prepare_int_xor = prepare_binary_int_op()
    prepare_int_lshift = prepare_binary_int_op()
    prepare_int_rshift = prepare_binary_int_op()
    prepare_uint_rshift = prepare_binary_int_op()
    prepare_uint_floordiv = prepare_binary_int_op()

    prepare_int_add_ovf = prepare_binary_int_op()
    prepare_int_sub_ovf = prepare_binary_int_op()
    prepare_int_mul_ovf = prepare_binary_int_op()

    prepare_int_neg = prepare_unary_int_op()
    prepare_int_invert = prepare_unary_int_op()

    prepare_int_le = prepare_cmp_op()
    prepare_int_lt = prepare_cmp_op()
    prepare_int_ge = prepare_cmp_op()
    prepare_int_gt = prepare_cmp_op()
    prepare_int_eq = prepare_cmp_op()
    prepare_int_ne = prepare_cmp_op()

    prepare_ptr_eq = prepare_int_eq
    prepare_ptr_ne = prepare_int_ne

    prepare_uint_lt = prepare_cmp_op()
    prepare_uint_le = prepare_cmp_op()
    prepare_uint_gt = prepare_cmp_op()
    prepare_uint_ge = prepare_cmp_op()

    prepare_int_is_true = prepare_unary_cmp()
    prepare_int_is_zero = prepare_unary_cmp()

    def prepare_finish(self, op):
        args = [locations.imm(self.frame_manager.frame_depth)]
        for i in range(op.numargs()):
            arg = op.getarg(i)
            if arg:
                args.append(self.loc(arg))
                self.possibly_free_var(arg)
            else:
                args.append(None)
        return args

    def _prepare_guard(self, op, args=None):
        if args is None:
            args = []
        args.append(imm(self.frame_manager.frame_depth))
        for arg in op.getfailargs():
            if arg:
                args.append(self.loc(arg))
            else:
                args.append(None)
        return args
    
    def prepare_guard_true(self, op):
        l0, box = self._ensure_value_is_boxed(op.getarg(0))
        args = self._prepare_guard(op, [l0])
        self.possibly_free_var(box)
        self.possibly_free_vars(op.getfailargs())
        return args

    prepare_guard_false = prepare_guard_true
    prepare_guard_nonnull = prepare_guard_true
    prepare_guard_isnull = prepare_guard_true

    def prepare_guard_no_overflow(self, op):
        locs = self._prepare_guard(op)
        self.possibly_free_vars(op.getfailargs())
        return locs

    prepare_guard_overflow = prepare_guard_no_overflow
    prepare_guard_not_invalidated = prepare_guard_no_overflow

    def prepare_guard_exception(self, op):
        boxes = list(op.getarglist())
        arg0 = ConstInt(rffi.cast(lltype.Signed, op.getarg(0).getint()))
        loc, box = self._ensure_value_is_boxed(arg0)
        boxes.append(box)
        box = TempInt()
        loc1 = self.force_allocate_reg(box, boxes)
        boxes.append(box)
        if op.result in self.longevity:
            resloc = self.force_allocate_reg(op.result, boxes)
            boxes.append(op.result)
        else:
            resloc = None
        pos_exc_value = imm(self.cpu.pos_exc_value())
        pos_exception = imm(self.cpu.pos_exception())
        arglocs = self._prepare_guard(op, [loc, loc1, resloc, pos_exc_value, pos_exception])
        self.possibly_free_vars(boxes)
        self.possibly_free_vars(op.getfailargs())
        return arglocs

    def prepare_guard_no_exception(self, op):
        loc, box = self._ensure_value_is_boxed(
                    ConstInt(self.cpu.pos_exception()))
        arglocs = self._prepare_guard(op, [loc])
        self.possibly_free_var(box)
        self.possibly_free_vars(op.getfailargs())
        return arglocs

    def prepare_guard_value(self, op):
        boxes = list(op.getarglist())
        b0, b1 = boxes
        imm_b1 = _check_imm_arg(b1)
        l0, box = self._ensure_value_is_boxed(b0, boxes)
        boxes.append(box)
        if not imm_b1:
            l1, box = self._ensure_value_is_boxed(b1,boxes)
            boxes.append(box)
        else:
            l1 = self.make_sure_var_in_reg(b1)
        assert op.result is None
        arglocs = self._prepare_guard(op, [l0, l1])
        self.possibly_free_vars(boxes)
        self.possibly_free_vars(op.getfailargs())
        return arglocs

    def prepare_guard_class(self, op):
        assert isinstance(op.getarg(0), Box)
        boxes = list(op.getarglist())

        x, x_box = self._ensure_value_is_boxed(boxes[0], boxes)
        boxes.append(x_box)

        t = TempInt()
        y = self.force_allocate_reg(t, boxes)
        boxes.append(t)
        y_val = rffi.cast(lltype.Signed, op.getarg(1).getint())
        self.assembler.mc.load_imm(y, y_val)

        offset = self.cpu.vtable_offset
        assert offset is not None
        offset_loc, offset_box = self._ensure_value_is_boxed(ConstInt(offset), boxes)
        boxes.append(offset_box)
        arglocs = self._prepare_guard(op, [x, y, offset_loc])
        self.possibly_free_vars(boxes)
        self.possibly_free_vars(op.getfailargs())
        return arglocs

    prepare_guard_nonnull_class = prepare_guard_class

    def prepare_guard_call_release_gil(self, op, guard_op):
        # first, close the stack in the sense of the asmgcc GC root tracker
        gcrootmap = self.cpu.gc_ll_descr.gcrootmap
        if gcrootmap:
            arglocs = []
            argboxes = []
            for i in range(op.numargs()):
                loc, box = self._ensure_value_is_boxed(op.getarg(i), argboxes)
                arglocs.append(loc)
                argboxes.append(box)
            self.assembler.call_release_gil(gcrootmap, arglocs, fcond)
            self.possibly_free_vars(argboxes)
        # do the call
        faildescr = guard_op.getdescr()
        fail_index = self.cpu.get_fail_descr_number(faildescr)
        self.assembler._write_fail_index(fail_index)
        args = [imm(rffi.cast(lltype.Signed, op.getarg(0).getint()))]
        self.assembler.emit_call(op, args, self, fail_index)
        # then reopen the stack
        if gcrootmap:
            self.assembler.call_reacquire_gil(gcrootmap, r.r0, fcond)
        locs = self._prepare_guard(guard_op)
        self.possibly_free_vars(guard_op.getfailargs())
        return locs

    def prepare_jump(self, op):
        descr = op.getdescr()
        assert isinstance(descr, LoopToken)
        nonfloatlocs = descr._ppc_arglocs[0]

        tmploc = r.r0       
        src_locs1 = [self.loc(op.getarg(i)) for i in range(op.numargs()) 
                            if op.getarg(i).type != FLOAT]
        assert tmploc not in nonfloatlocs
        dst_locs1 = [loc for loc in nonfloatlocs if loc is not None]
        remap_frame_layout_mixed(self.assembler,
                                 src_locs1, dst_locs1, tmploc,
                                 [], [], None)
        return []

    def prepare_setfield_gc(self, op):
        boxes = list(op.getarglist())
        b0, b1 = boxes
        ofs, size, ptr = self._unpack_fielddescr(op.getdescr())
        base_loc, base_box = self._ensure_value_is_boxed(b0, boxes)
        boxes.append(base_box)
        value_loc, value_box = self._ensure_value_is_boxed(b1, boxes)
        boxes.append(value_box)
        c_ofs = ConstInt(ofs)
        if _check_imm_arg(c_ofs):
            ofs_loc = imm(ofs)
        else:
            ofs_loc, ofs_box = self._ensure_value_is_boxed(c_ofs, boxes)
            boxes.append(ofs_box)
        self.possibly_free_vars(boxes)
        return [value_loc, base_loc, ofs_loc, imm(size)]

    prepare_setfield_raw = prepare_setfield_gc

    def prepare_getfield_gc(self, op):
        a0 = op.getarg(0)
        ofs, size, ptr = self._unpack_fielddescr(op.getdescr())
        base_loc, base_box = self._ensure_value_is_boxed(a0)
        c_ofs = ConstInt(ofs)
        if _check_imm_arg(c_ofs):
            ofs_loc = imm(ofs)
        else:
            ofs_loc, ofs_box = self._ensure_value_is_boxed(c_ofs, [base_box])
            self.possibly_free_var(ofs_box)
        self.possibly_free_var(a0)
        self.possibly_free_var(base_box)
        res = self.force_allocate_reg(op.result)
        self.possibly_free_var(op.result)
        return [base_loc, ofs_loc, res, imm(size)]

    prepare_getfield_raw = prepare_getfield_gc
    prepare_getfield_raw_pure = prepare_getfield_gc
    prepare_getfield_gc_pure = prepare_getfield_gc

    def prepare_arraylen_gc(self, op):
        arraydescr = op.getdescr()
        assert isinstance(arraydescr, BaseArrayDescr)
        ofs = arraydescr.get_ofs_length(self.cpu.translate_support_code)
        arg = op.getarg(0)
        base_loc, base_box = self._ensure_value_is_boxed(arg)
        self.possibly_free_vars([arg, base_box])

        res = self.force_allocate_reg(op.result)
        self.possibly_free_var(op.result)
        return [res, base_loc, imm(ofs)]

    def prepare_setarrayitem_gc(self, op):
        b0, b1, b2 = boxes = list(op.getarglist())
        _, scale, ofs, _, ptr = self._unpack_arraydescr(op.getdescr())

        base_loc, base_box  = self._ensure_value_is_boxed(b0, boxes)
        boxes.append(base_box)
        ofs_loc, ofs_box = self._ensure_value_is_boxed(b1, boxes)
        boxes.append(ofs_box)
        #XXX check if imm would be fine here
        value_loc, value_box = self._ensure_value_is_boxed(b2, boxes)
        boxes.append(value_box)
        if scale > 0:
            tmp, box = self.allocate_scratch_reg(forbidden_vars=boxes)
            boxes.append(box)
        else:
            tmp = None
        self.possibly_free_vars(boxes)
        return [value_loc, base_loc, ofs_loc, imm(scale), imm(ofs), tmp]

    prepare_setarrayitem_raw = prepare_setarrayitem_gc

    def prepare_getarrayitem_gc(self, op):
        a0, a1 = boxes = list(op.getarglist())
        _, scale, ofs, _, ptr = self._unpack_arraydescr(op.getdescr())
        base_loc, base_box  = self._ensure_value_is_boxed(a0, boxes)
        boxes.append(base_box)
        ofs_loc, ofs_box = self._ensure_value_is_boxed(a1, boxes)
        boxes.append(ofs_box)
        if scale > 0:
            tmp, box = self.allocate_scratch_reg(forbidden_vars=boxes)
            boxes.append(box)
        else:
            tmp = None
        self.possibly_free_vars(boxes)
        res = self.force_allocate_reg(op.result)
        self.possibly_free_var(op.result)
        return [res, base_loc, ofs_loc, imm(scale), imm(ofs), tmp]

    prepare_getarrayitem_raw = prepare_getarrayitem_gc
    prepare_getarrayitem_gc_pure = prepare_getarrayitem_gc

    def prepare_strlen(self, op):
        l0, box = self._ensure_value_is_boxed(op.getarg(0))
        boxes = [box]

        basesize, itemsize, ofs_length = symbolic.get_array_token(rstr.STR,
                                         self.cpu.translate_support_code)
        ofs_box = ConstInt(ofs_length)
        imm_ofs = _check_imm_arg(ofs_box)

        if imm_ofs:
            l1 = self.make_sure_var_in_reg(ofs_box, boxes)
        else:
            l1, box1 = self._ensure_value_is_boxed(ofs_box, boxes)
            boxes.append(box1)

        self.possibly_free_vars(boxes)
        res = self.force_allocate_reg(op.result)
        self.possibly_free_var(op.result)
        return [l0, l1, res]

    def prepare_strgetitem(self, op):
        boxes = list(op.getarglist())
        base_loc, box = self._ensure_value_is_boxed(boxes[0])
        boxes.append(box)

        a1 = boxes[1]
        imm_a1 = _check_imm_arg(a1)
        if imm_a1:
            ofs_loc = self.make_sure_var_in_reg(a1, boxes)
        else:
            ofs_loc, box = self._ensure_value_is_boxed(a1, boxes)
            boxes.append(box)

        self.possibly_free_vars(boxes)
        res = self.force_allocate_reg(op.result)
        self.possibly_free_var(op.result)

        basesize, itemsize, ofs_length = symbolic.get_array_token(rstr.STR,
                                         self.cpu.translate_support_code)
        assert itemsize == 1
        return [res, base_loc, ofs_loc, imm(basesize)]

    def prepare_strsetitem(self, op):
        boxes = list(op.getarglist())

        base_loc, box = self._ensure_value_is_boxed(boxes[0], boxes)
        boxes.append(box)

        ofs_loc, box = self._ensure_value_is_boxed(boxes[1], boxes)
        boxes.append(box)

        value_loc, box = self._ensure_value_is_boxed(boxes[2], boxes)
        boxes.append(box)

        self.possibly_free_vars(boxes)

        basesize, itemsize, ofs_length = symbolic.get_array_token(rstr.STR,
                                         self.cpu.translate_support_code)
        assert itemsize == 1
        return [value_loc, base_loc, ofs_loc, imm(basesize)]

    prepare_copystrcontent = void
    prepare_copyunicodecontent = void

    def prepare_unicodelen(self, op):
        l0, box = self._ensure_value_is_boxed(op.getarg(0))
        boxes = [box]
        basesize, itemsize, ofs_length = symbolic.get_array_token(rstr.UNICODE,
                                         self.cpu.translate_support_code)
        ofs_box = ConstInt(ofs_length)
        imm_ofs = _check_imm_arg(ofs_box)

        if imm_ofs:
            l1 = imm(ofs_length)
        else:
            l1, box1 = self._ensure_value_is_boxed(ofs_box, boxes)
            boxes.append(box1)

        self.possibly_free_vars(boxes)
        res = self.force_allocate_reg(op.result)
        self.possibly_free_var(op.result)
        return [l0, l1, res]

    def prepare_unicodegetitem(self, op):
        boxes = list(op.getarglist())
        base_loc, box = self._ensure_value_is_boxed(boxes[0], boxes)
        boxes.append(box)
        ofs_loc, box = self._ensure_value_is_boxed(boxes[1], boxes)
        boxes.append(box)
        self.possibly_free_vars(boxes)

        res = self.force_allocate_reg(op.result)
        self.possibly_free_var(op.result)

        basesize, itemsize, ofs_length = symbolic.get_array_token(rstr.UNICODE,
                                         self.cpu.translate_support_code)
        scale = itemsize/2
        return [res, base_loc, ofs_loc, imm(scale), imm(basesize), imm(itemsize)]

    def prepare_unicodesetitem(self, op):
        boxes = list(op.getarglist())
        base_loc, box = self._ensure_value_is_boxed(boxes[0], boxes)
        boxes.append(box)
        ofs_loc, box = self._ensure_value_is_boxed(boxes[1], boxes)
        boxes.append(box)
        value_loc, box = self._ensure_value_is_boxed(boxes[2], boxes)
        boxes.append(box)

        self.possibly_free_vars(boxes)

        basesize, itemsize, ofs_length = symbolic.get_array_token(rstr.UNICODE,
                                         self.cpu.translate_support_code)
        scale = itemsize/2
        return [value_loc, base_loc, ofs_loc, imm(scale), imm(basesize), imm(itemsize)]

    def prepare_same_as(self, op):
        arg = op.getarg(0)
        imm_arg = _check_imm_arg(arg)
        if imm_arg:
            argloc = self.make_sure_var_in_reg(arg)
        else:
            argloc, box = self._ensure_value_is_boxed(arg)
            self.possibly_free_var(box)

        resloc = self.force_allocate_reg(op.result)
        self.possibly_free_var(op.result)
        return [argloc, resloc]

    prepare_cast_ptr_to_int = prepare_same_as
    prepare_cast_int_to_ptr = prepare_same_as

    def prepare_new(self, op):
        gc_ll_descr = self.assembler.cpu.gc_ll_descr
        # XXX introduce the fastpath for malloc
        arglocs = self._prepare_args_for_new_op(op.getdescr())
        force_index = self.assembler.write_new_force_index()
        self.assembler._emit_call(force_index, self.assembler.malloc_func_addr,
                                arglocs, self, result=op.result)
        self.possibly_free_vars(arglocs)
        self.possibly_free_var(op.result)
        return []

    def prepare_new_with_vtable(self, op):
        classint = op.getarg(0).getint()
        descrsize = heaptracker.vtable2descr(self.cpu, classint)
        # XXX add fastpath for allocation
        callargs = self._prepare_args_for_new_op(descrsize)
        force_index = self.assembler.write_new_force_index()
        self.assembler._emit_call(force_index, self.assembler.malloc_func_addr,
                                    callargs, self, result=op.result)
        self.possibly_free_vars(callargs)
        self.possibly_free_var(op.result)
        return [imm(classint)]

    def prepare_new_array(self, op):
        gc_ll_descr = self.cpu.gc_ll_descr
        if gc_ll_descr.get_funcptr_for_newarray is not None:
            # framework GC
            box_num_elem = op.getarg(0)
            if isinstance(box_num_elem, ConstInt):
                num_elem = box_num_elem.value
                # XXX implement fastpath for malloc
            args = self.assembler.cpu.gc_ll_descr.args_for_new_array(
                op.getdescr())
            argboxes = [ConstInt(x) for x in args]
            argboxes.append(box_num_elem)
            force_index = self.assembler.write_new_force_index()
            self.assembler._emit_call(force_index, self.assembler.malloc_array_func_addr,
                                        argboxes, self, result=op.result)
            return []
        # boehm GC
        itemsize, scale, basesize, ofs_length, _ = (
            self._unpack_arraydescr(op.getdescr()))
        return self._malloc_varsize(basesize, ofs_length, itemsize, op)

    def prepare_newstr(self, op):
        gc_ll_descr = self.cpu.gc_ll_descr
        if gc_ll_descr.get_funcptr_for_newstr is not None:
            force_index = self.assembler.write_new_force_index()
            self.assembler._emit_call(force_index,
                    self.assembler.malloc_str_func_addr, [op.getarg(0)],
                    self, op.result)
            return []
        # boehm GC
        ofs_items, itemsize, ofs = symbolic.get_array_token(rstr.STR,
                            self.cpu.translate_support_code)
        assert itemsize == 1
        return self._malloc_varsize(ofs_items, ofs, itemsize, op)

    def prepare_newunicode(self, op):
        gc_ll_descr = self.cpu.gc_ll_descr
        if gc_ll_descr.get_funcptr_for_newunicode is not None:
            force_index = self.assembler.write_new_force_index()
            self.assembler._emit_call(force_index, 
                    self.assembler.malloc_unicode_func_addr,
                    [op.getarg(0)], self, op.result)
            return []
        # boehm GC
        ofs_items, _, ofs = symbolic.get_array_token(rstr.UNICODE,
                            self.cpu.translate_support_code)
        _, itemsize, _ = symbolic.get_array_token(rstr.UNICODE,
                            self.cpu.translate_support_code)
        return self._malloc_varsize(ofs_items, ofs, itemsize, op)

    def prepare_call(self, op):
        effectinfo = op.getdescr().get_extra_info()
        if effectinfo is not None:
            oopspecindex = effectinfo.oopspecindex
            if oopspecindex == EffectInfo.OS_MATH_SQRT:
                args = self.prepare_op_math_sqrt(op, fcond)
                self.assembler.emit_op_math_sqrt(op, args, self, fcond)
                return
        args = [imm(rffi.cast(lltype.Signed, op.getarg(0).getint()))]
        return args

    prepare_debug_merge_point = void
    prepare_jit_debug = void

    def prepare_cond_call_gc_wb(self, op):
        assert op.result is None
        N = op.numargs()
        # we force all arguments in a reg (unless they are Consts),
        # because it will be needed anyway by the following setfield_gc
        # or setarrayitem_gc. It avoids loading it twice from the memory.
        arglocs = []
        argboxes = []
        for i in range(N):
            loc, box = self._ensure_value_is_boxed(op.getarg(i), argboxes)
            arglocs.append(loc)
            argboxes.append(box)
        self.rm.possibly_free_vars(argboxes)
        return arglocs

    prepare_cond_call_gc_wb_array = prepare_cond_call_gc_wb

    def prepare_force_token(self, op):
        res_loc = self.force_allocate_reg(op.result)
        self.possibly_free_var(op.result)
        return [res_loc]

    def prepare_guard_call_may_force(self, op, guard_op):
        faildescr = guard_op.getdescr()
        fail_index = self.cpu.get_fail_descr_number(faildescr)
        self.assembler._write_fail_index(fail_index)
        args = [imm(rffi.cast(lltype.Signed, op.getarg(0).getint()))]
        for v in guard_op.getfailargs():
            if v in self.rm.reg_bindings:
                self.force_spill_var(v)
        self.assembler.emit_call(op, args, self, fail_index)
        locs = self._prepare_guard(guard_op)
        self.possibly_free_vars(guard_op.getfailargs())
        return locs

    def prepare_guard_call_assembler(self, op, guard_op):
        descr = op.getdescr()
        assert isinstance(descr, LoopToken)
        jd = descr.outermost_jitdriver_sd
        assert jd is not None
        size = jd.portal_calldescr.get_result_size(self.cpu.translate_support_code)
        vable_index = jd.index_of_virtualizable
        if vable_index >= 0:
            self._sync_var(op.getarg(vable_index))
            vable = self.frame_manager.loc(op.getarg(vable_index))
        else:
            vable = imm(0)
        self.possibly_free_vars(guard_op.getfailargs())
        return [imm(size), vable]

    def _prepare_args_for_new_op(self, new_args):
        gc_ll_descr = self.cpu.gc_ll_descr
        args = gc_ll_descr.args_for_new(new_args)
        arglocs = []
        for i in range(len(args)):
            arg = args[i]
            t = TempInt()
            l = self.force_allocate_reg(t, selected_reg=r.MANAGED_REGS[i])
            self.assembler.load(l, imm(arg))
            arglocs.append(t)
        return arglocs

    def _malloc_varsize(self, ofs_items, ofs_length, itemsize, op):
        v = op.getarg(0)
        res_v = op.result
        boxes = [v, res_v]
        itemsize_box = ConstInt(itemsize)
        ofs_items_box = ConstInt(ofs_items)
        if _check_imm_arg(ofs_items_box):
            ofs_items_loc = self.convert_to_imm(ofs_items_box)
        else:
            ofs_items_loc, ofs_items_box = self._ensure_value_is_boxed(ofs_items_box, boxes)
            boxes.append(ofs_items_box)
        vloc, vbox = self._ensure_value_is_boxed(v, [res_v])
        boxes.append(vbox)
        size, size_box = self._ensure_value_is_boxed(itemsize_box, boxes)
        boxes.append(size_box)
        self.assembler._regalloc_malloc_varsize(size, size_box,
                                vloc, vbox, ofs_items_loc, self, res_v)
        base_loc = self.make_sure_var_in_reg(res_v)

        value_loc, vbox = self._ensure_value_is_boxed(v, [res_v])
        boxes.append(vbox)
        self.possibly_free_vars(boxes)
        assert value_loc.is_reg()
        assert base_loc.is_reg()
        return [value_loc, base_loc, imm(ofs_length)]

    # from ../x86/regalloc.py:791
    def _unpack_fielddescr(self, fielddescr):
        assert isinstance(fielddescr, BaseFieldDescr)
        ofs = fielddescr.offset
        size = fielddescr.get_field_size(self.cpu.translate_support_code)
        ptr = fielddescr.is_pointer_field()
        return ofs, size, ptr

    # from ../x86/regalloc.py:779
    def _unpack_arraydescr(self, arraydescr):
        assert isinstance(arraydescr, BaseArrayDescr)
        cpu = self.cpu
        ofs_length = arraydescr.get_ofs_length(cpu.translate_support_code)
        ofs = arraydescr.get_base_size(cpu.translate_support_code)
        size = arraydescr.get_item_size(cpu.translate_support_code)
        ptr = arraydescr.is_array_of_pointers()
        scale = 0
        while (1 << scale) < size:
            scale += 1
        assert (1 << scale) == size
        return size, scale, ofs, ofs_length, ptr

    def prepare_force_spill(self, op):
        self.force_spill_var(op.getarg(0))
        return []

def add_none_argument(fn):
    return lambda self, op: fn(self, op, None)

def notimplemented(self, op):
    raise NotImplementedError, op

def notimplemented_with_guard(self, op, guard_op):

    raise NotImplementedError, op

operations = [notimplemented] * (rop._LAST + 1)
operations_with_guard = [notimplemented_with_guard] * (rop._LAST + 1)

for key, value in rop.__dict__.items():
    key = key.lower()
    if key.startswith('_'):
        continue
    methname = 'prepare_%s' % key
    if hasattr(Regalloc, methname):
        func = getattr(Regalloc, methname).im_func
        operations[value] = func

for key, value in rop.__dict__.items():
    key = key.lower()
    if key.startswith('_'):
        continue
    methname = 'prepare_guard_%s' % key
    if hasattr(Regalloc, methname):
        func = getattr(Regalloc, methname).im_func
        operations_with_guard[value] = func
        operations[value] = add_none_argument(func)

Regalloc.operations = operations
Regalloc.operations_with_guard = operations_with_guard
