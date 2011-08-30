import py, random

from pypy.rpython.lltypesystem import lltype, llmemory, rclass, rstr
from pypy.rpython.ootypesystem import ootype
from pypy.rpython.lltypesystem.rclass import OBJECT, OBJECT_VTABLE
from pypy.rpython.rclass import FieldListAccessor, IR_QUASIIMMUTABLE

from pypy.jit.backend.llgraph import runner
from pypy.jit.metainterp.history import (BoxInt, BoxPtr, ConstInt, ConstPtr,
                                         Const, TreeLoop, BoxObj,
                                         ConstObj, AbstractDescr)
from pypy.jit.metainterp.optimizeopt.util import sort_descrs, equaloplists
from pypy.jit.metainterp.optimize import InvalidLoop
from pypy.jit.codewriter.effectinfo import EffectInfo
from pypy.jit.codewriter.heaptracker import register_known_gctype, adr2int
from pypy.jit.tool.oparser import parse, pure_parse
from pypy.jit.metainterp.quasiimmut import QuasiImmutDescr
from pypy.jit.metainterp import compile, resume, history
from pypy.jit.metainterp.jitprof import EmptyProfiler
from pypy.config.pypyoption import get_pypy_config

def test_sort_descrs():
    class PseudoDescr(AbstractDescr):
        def __init__(self, n):
            self.n = n
        def sort_key(self):
            return self.n
    for i in range(17):
        lst = [PseudoDescr(j) for j in range(i)]
        lst2 = lst[:]
        random.shuffle(lst2)
        sort_descrs(lst2)
        assert lst2 == lst

def test_equaloplists():
    ops = """
    [i0]
    i1 = int_add(i0, 1)
    i2 = int_add(i1, 1)
    guard_true(i1) [i2]
    jump(i1)
    """
    namespace = {}
    loop1 = pure_parse(ops, namespace=namespace)
    loop2 = pure_parse(ops, namespace=namespace)
    loop3 = pure_parse(ops.replace("i2 = int_add", "i2 = int_sub"),
                       namespace=namespace)
    assert equaloplists(loop1.operations, loop2.operations)
    py.test.raises(AssertionError,
                   "equaloplists(loop1.operations, loop3.operations)")

def test_equaloplists_fail_args():
    ops = """
    [i0]
    i1 = int_add(i0, 1)
    i2 = int_add(i1, 1)
    guard_true(i1) [i2, i1]
    jump(i1)
    """
    namespace = {}
    loop1 = pure_parse(ops, namespace=namespace)
    loop2 = pure_parse(ops.replace("[i2, i1]", "[i1, i2]"),
                       namespace=namespace)
    py.test.raises(AssertionError,
                   "equaloplists(loop1.operations, loop2.operations)")
    assert equaloplists(loop1.operations, loop2.operations,
                        strict_fail_args=False)
    loop3 = pure_parse(ops.replace("[i2, i1]", "[i2, i0]"),
                       namespace=namespace)
    py.test.raises(AssertionError,
                   "equaloplists(loop1.operations, loop3.operations)")

# ____________________________________________________________

class LLtypeMixin(object):
    type_system = 'lltype'

    def get_class_of_box(self, box):
        return box.getref(rclass.OBJECTPTR).typeptr

    node_vtable = lltype.malloc(OBJECT_VTABLE, immortal=True)
    node_vtable.name = rclass.alloc_array_name('node')
    node_vtable_adr = llmemory.cast_ptr_to_adr(node_vtable)
    node_vtable2 = lltype.malloc(OBJECT_VTABLE, immortal=True)
    node_vtable2.name = rclass.alloc_array_name('node2')
    node_vtable_adr2 = llmemory.cast_ptr_to_adr(node_vtable2)
    cpu = runner.LLtypeCPU(None)

    NODE = lltype.GcForwardReference()
    NODE.become(lltype.GcStruct('NODE', ('parent', OBJECT),
                                        ('value', lltype.Signed),
                                        ('floatval', lltype.Float),
                                        ('next', lltype.Ptr(NODE))))
    NODE2 = lltype.GcStruct('NODE2', ('parent', NODE),
                                     ('other', lltype.Ptr(NODE)))
    node = lltype.malloc(NODE)
    node.parent.typeptr = node_vtable
    node2 = lltype.malloc(NODE2)
    node2.parent.parent.typeptr = node_vtable2
    nodebox = BoxPtr(lltype.cast_opaque_ptr(llmemory.GCREF, node))
    myptr = nodebox.value
    myptr2 = lltype.cast_opaque_ptr(llmemory.GCREF, lltype.malloc(NODE))
    nullptr = lltype.nullptr(llmemory.GCREF.TO)
    nodebox2 = BoxPtr(lltype.cast_opaque_ptr(llmemory.GCREF, node2))
    nodesize = cpu.sizeof(NODE)
    nodesize2 = cpu.sizeof(NODE2)
    valuedescr = cpu.fielddescrof(NODE, 'value')
    floatdescr = cpu.fielddescrof(NODE, 'floatval')
    nextdescr = cpu.fielddescrof(NODE, 'next')
    otherdescr = cpu.fielddescrof(NODE2, 'other')

    accessor = FieldListAccessor()
    accessor.initialize(None, {'inst_field': IR_QUASIIMMUTABLE})
    QUASI = lltype.GcStruct('QUASIIMMUT', ('inst_field', lltype.Signed),
                            ('mutate_field', rclass.OBJECTPTR),
                            hints={'immutable_fields': accessor})
    quasisize = cpu.sizeof(QUASI)
    quasi = lltype.malloc(QUASI, immortal=True)
    quasi.inst_field = -4247
    quasifielddescr = cpu.fielddescrof(QUASI, 'inst_field')
    quasibox = BoxPtr(lltype.cast_opaque_ptr(llmemory.GCREF, quasi))
    quasiimmutdescr = QuasiImmutDescr(cpu, quasibox,
                                      quasifielddescr,
                                      cpu.fielddescrof(QUASI, 'mutate_field'))

    NODEOBJ = lltype.GcStruct('NODEOBJ', ('parent', OBJECT),
                                         ('ref', lltype.Ptr(OBJECT)))
    nodeobj = lltype.malloc(NODEOBJ)
    nodeobjvalue = lltype.cast_opaque_ptr(llmemory.GCREF, nodeobj)
    refdescr = cpu.fielddescrof(NODEOBJ, 'ref')

    INTOBJ_NOIMMUT = lltype.GcStruct('INTOBJ_NOIMMUT', ('parent', OBJECT),
                                                ('intval', lltype.Signed))
    INTOBJ_IMMUT = lltype.GcStruct('INTOBJ_IMMUT', ('parent', OBJECT),
                                            ('intval', lltype.Signed),
                                            hints={'immutable': True})
    intobj_noimmut_vtable = lltype.malloc(OBJECT_VTABLE, immortal=True)
    intobj_immut_vtable = lltype.malloc(OBJECT_VTABLE, immortal=True)
    noimmut_intval = cpu.fielddescrof(INTOBJ_NOIMMUT, 'intval')
    immut_intval = cpu.fielddescrof(INTOBJ_IMMUT, 'intval')

    arraydescr = cpu.arraydescrof(lltype.GcArray(lltype.Signed))
    floatarraydescr = cpu.arraydescrof(lltype.GcArray(lltype.Float))

    # a GcStruct not inheriting from OBJECT
    S = lltype.GcStruct('TUPLE', ('a', lltype.Signed), ('b', lltype.Ptr(NODE)))
    ssize = cpu.sizeof(S)
    adescr = cpu.fielddescrof(S, 'a')
    bdescr = cpu.fielddescrof(S, 'b')
    sbox = BoxPtr(lltype.cast_opaque_ptr(llmemory.GCREF, lltype.malloc(S)))
    arraydescr2 = cpu.arraydescrof(lltype.GcArray(lltype.Ptr(S)))

    T = lltype.GcStruct('TUPLE',
                        ('c', lltype.Signed),
                        ('d', lltype.Ptr(lltype.GcArray(lltype.Ptr(NODE)))))
    tsize = cpu.sizeof(T)
    cdescr = cpu.fielddescrof(T, 'c')
    ddescr = cpu.fielddescrof(T, 'd')
    arraydescr3 = cpu.arraydescrof(lltype.GcArray(lltype.Ptr(NODE)))

    U = lltype.GcStruct('U',
                        ('parent', OBJECT),
                        ('one', lltype.Ptr(lltype.GcArray(lltype.Ptr(NODE)))))
    u_vtable = lltype.malloc(OBJECT_VTABLE, immortal=True)
    u_vtable_adr = llmemory.cast_ptr_to_adr(u_vtable)
    usize = cpu.sizeof(U)
    onedescr = cpu.fielddescrof(U, 'one')

    FUNC = lltype.FuncType([lltype.Signed], lltype.Signed)
    plaincalldescr = cpu.calldescrof(FUNC, FUNC.ARGS, FUNC.RESULT,
                                     EffectInfo.MOST_GENERAL)
    nonwritedescr = cpu.calldescrof(FUNC, FUNC.ARGS, FUNC.RESULT,
                                    EffectInfo([], [], [], []))
    writeadescr = cpu.calldescrof(FUNC, FUNC.ARGS, FUNC.RESULT,
                                  EffectInfo([], [], [adescr], []))
    writearraydescr = cpu.calldescrof(FUNC, FUNC.ARGS, FUNC.RESULT,
                                  EffectInfo([], [], [adescr], [arraydescr]))
    readadescr = cpu.calldescrof(FUNC, FUNC.ARGS, FUNC.RESULT,
                                 EffectInfo([adescr], [], [], []))
    mayforcevirtdescr = cpu.calldescrof(FUNC, FUNC.ARGS, FUNC.RESULT,
                 EffectInfo([nextdescr], [], [], [],
                            EffectInfo.EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE,
                            can_invalidate=True))
    arraycopydescr = cpu.calldescrof(FUNC, FUNC.ARGS, FUNC.RESULT,
             EffectInfo([], [], [], [], oopspecindex=EffectInfo.OS_ARRAYCOPY))

    for _name, _os in [
        ('strconcatdescr',               'OS_STR_CONCAT'),
        ('strslicedescr',                'OS_STR_SLICE'),
        ('strequaldescr',                'OS_STR_EQUAL'),
        ('streq_slice_checknull_descr',  'OS_STREQ_SLICE_CHECKNULL'),
        ('streq_slice_nonnull_descr',    'OS_STREQ_SLICE_NONNULL'),
        ('streq_slice_char_descr',       'OS_STREQ_SLICE_CHAR'),
        ('streq_nonnull_descr',          'OS_STREQ_NONNULL'),
        ('streq_nonnull_char_descr',     'OS_STREQ_NONNULL_CHAR'),
        ('streq_checknull_char_descr',   'OS_STREQ_CHECKNULL_CHAR'),
        ('streq_lengthok_descr',         'OS_STREQ_LENGTHOK'),
        ]:
        _oopspecindex = getattr(EffectInfo, _os)
        locals()[_name] = \
            cpu.calldescrof(FUNC, FUNC.ARGS, FUNC.RESULT,
                EffectInfo([], [], [], [], oopspecindex=_oopspecindex))
        #
        _oopspecindex = getattr(EffectInfo, _os.replace('STR', 'UNI'))
        locals()[_name.replace('str', 'unicode')] = \
            cpu.calldescrof(FUNC, FUNC.ARGS, FUNC.RESULT,
                EffectInfo([], [], [], [], oopspecindex=_oopspecindex))

    s2u_descr = cpu.calldescrof(FUNC, FUNC.ARGS, FUNC.RESULT,
            EffectInfo([], [], [], [], oopspecindex=EffectInfo.OS_STR2UNICODE))
    #

    class LoopToken(AbstractDescr):
        pass
    asmdescr = LoopToken() # it can be whatever, it's not a descr though

    from pypy.jit.metainterp.virtualref import VirtualRefInfo
    class FakeWarmRunnerDesc:
        pass
    FakeWarmRunnerDesc.cpu = cpu
    vrefinfo = VirtualRefInfo(FakeWarmRunnerDesc)
    virtualtokendescr = vrefinfo.descr_virtual_token
    virtualforceddescr = vrefinfo.descr_forced
    jit_virtual_ref_vtable = vrefinfo.jit_virtual_ref_vtable
    jvr_vtable_adr = llmemory.cast_ptr_to_adr(jit_virtual_ref_vtable)

    register_known_gctype(cpu, node_vtable,  NODE)
    register_known_gctype(cpu, node_vtable2, NODE2)
    register_known_gctype(cpu, u_vtable,     U)
    register_known_gctype(cpu, jit_virtual_ref_vtable,vrefinfo.JIT_VIRTUAL_REF)
    register_known_gctype(cpu, intobj_noimmut_vtable, INTOBJ_NOIMMUT)
    register_known_gctype(cpu, intobj_immut_vtable,   INTOBJ_IMMUT)

    namespace = locals()

class OOtypeMixin_xxx_disabled(object):
    type_system = 'ootype'

##    def get_class_of_box(self, box):
##        root = box.getref(ootype.ROOT)
##        return ootype.classof(root)
    
##    cpu = runner.OOtypeCPU(None)
##    NODE = ootype.Instance('NODE', ootype.ROOT, {})
##    NODE._add_fields({'value': ootype.Signed,
##                      'floatval' : ootype.Float,
##                      'next': NODE})
##    NODE2 = ootype.Instance('NODE2', NODE, {'other': NODE})

##    node_vtable = ootype.runtimeClass(NODE)
##    node_vtable_adr = ootype.cast_to_object(node_vtable)
##    node_vtable2 = ootype.runtimeClass(NODE2)
##    node_vtable_adr2 = ootype.cast_to_object(node_vtable2)

##    node = ootype.new(NODE)
##    nodebox = BoxObj(ootype.cast_to_object(node))
##    myptr = nodebox.value
##    myptr2 = ootype.cast_to_object(ootype.new(NODE))
##    nodebox2 = BoxObj(ootype.cast_to_object(node))
##    valuedescr = cpu.fielddescrof(NODE, 'value')
##    floatdescr = cpu.fielddescrof(NODE, 'floatval')
##    nextdescr = cpu.fielddescrof(NODE, 'next')
##    otherdescr = cpu.fielddescrof(NODE2, 'other')
##    nodesize = cpu.typedescrof(NODE)
##    nodesize2 = cpu.typedescrof(NODE2)

##    arraydescr = cpu.arraydescrof(ootype.Array(ootype.Signed))
##    floatarraydescr = cpu.arraydescrof(ootype.Array(ootype.Float))

##    # a plain Record
##    S = ootype.Record({'a': ootype.Signed, 'b': NODE})
##    ssize = cpu.typedescrof(S)
##    adescr = cpu.fielddescrof(S, 'a')
##    bdescr = cpu.fielddescrof(S, 'b')
##    sbox = BoxObj(ootype.cast_to_object(ootype.new(S)))
##    arraydescr2 = cpu.arraydescrof(ootype.Array(S))

##    T = ootype.Record({'c': ootype.Signed,
##                       'd': ootype.Array(NODE)})
##    tsize = cpu.typedescrof(T)
##    cdescr = cpu.fielddescrof(T, 'c')
##    ddescr = cpu.fielddescrof(T, 'd')
##    arraydescr3 = cpu.arraydescrof(ootype.Array(NODE))

##    U = ootype.Instance('U', ootype.ROOT, {'one': ootype.Array(NODE)})
##    usize = cpu.typedescrof(U)
##    onedescr = cpu.fielddescrof(U, 'one')
##    u_vtable = ootype.runtimeClass(U)
##    u_vtable_adr = ootype.cast_to_object(u_vtable)

##    # force a consistent order
##    valuedescr.sort_key()
##    nextdescr.sort_key()
##    adescr.sort_key()
##    bdescr.sort_key()

##    FUNC = lltype.FuncType([lltype.Signed], lltype.Signed)
##    nonwritedescr = cpu.calldescrof(FUNC, FUNC.ARGS, FUNC.RESULT) # XXX fix ootype

##    cpu.class_sizes = {node_vtable_adr: cpu.typedescrof(NODE),
##                       node_vtable_adr2: cpu.typedescrof(NODE2),
##                       u_vtable_adr: cpu.typedescrof(U)}
##    namespace = locals()

# ____________________________________________________________



class Fake(object):
    failargs_limit = 1000
    storedebug = None


class FakeMetaInterpStaticData(object):

    def __init__(self, cpu):
        self.cpu = cpu
        self.profiler = EmptyProfiler()
        self.options = Fake()
        self.globaldata = Fake()
        self.config = get_pypy_config(translating=True)
        self.config.translation.jit_ffi = True

    class warmrunnerdesc:
        class memory_manager:
            retrace_limit = 5
            max_retrace_guards = 15

class Storage(compile.ResumeGuardDescr):
    "for tests."
    def __init__(self, metainterp_sd=None, original_greenkey=None):
        self.metainterp_sd = metainterp_sd
        self.original_greenkey = original_greenkey
    def store_final_boxes(self, op, boxes):
        op.setfailargs(boxes)
    def __eq__(self, other):
        return type(self) is type(other)      # xxx obscure
    def clone_if_mutable(self):
        res = Storage(self.metainterp_sd, self.original_greenkey)
        self.copy_all_attributes_into(res)
        return res

def _sortboxes(boxes):
    _kind2count = {history.INT: 1, history.REF: 2, history.FLOAT: 3}
    return sorted(boxes, key=lambda box: _kind2count[box.type])

class BaseTest(object):

    def parse(self, s, boxkinds=None):
        return parse(s, self.cpu, self.namespace,
                     type_system=self.type_system,
                     boxkinds=boxkinds,
                     invent_fail_descr=self.invent_fail_descr)

    def invent_fail_descr(self, model, fail_args):
        if fail_args is None:
            return None
        descr = Storage()
        descr.rd_frame_info_list = resume.FrameInfo(None, "code", 11)
        descr.rd_snapshot = resume.Snapshot(None, _sortboxes(fail_args))
        return descr

    def assert_equal(self, optimized, expected, text_right=None):
        from pypy.jit.metainterp.optimizeopt.util import equaloplists
        assert len(optimized.inputargs) == len(expected.inputargs)
        remap = {}
        for box1, box2 in zip(optimized.inputargs, expected.inputargs):
            assert box1.__class__ == box2.__class__
            remap[box2] = box1
        assert equaloplists(optimized.operations,
                            expected.operations, False, remap, text_right)

    def _do_optimize_loop(self, loop, call_pure_results):
        from pypy.jit.metainterp.optimizeopt import optimize_loop_1
        from pypy.jit.metainterp.optimizeopt.util import args_dict

        self.loop = loop
        loop.call_pure_results = args_dict()
        if call_pure_results is not None:
            for k, v in call_pure_results.items():
                loop.call_pure_results[list(k)] = v
        metainterp_sd = FakeMetaInterpStaticData(self.cpu)
        if hasattr(self, 'vrefinfo'):
            metainterp_sd.virtualref_info = self.vrefinfo
        if hasattr(self, 'callinfocollection'):
            metainterp_sd.callinfocollection = self.callinfocollection
        #
        optimize_loop_1(metainterp_sd, loop, self.enable_opts)

# ____________________________________________________________

