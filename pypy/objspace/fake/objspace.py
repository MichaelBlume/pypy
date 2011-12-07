from pypy.interpreter.baseobjspace import W_Root, ObjSpace, SpaceCache
from pypy.interpreter import argument, gateway
from pypy.interpreter.typedef import TypeDef
from pypy.annotation.model import SomeInstance, s_None
from pypy.rpython.extregistry import ExtRegistryEntry
from pypy.rpython.lltypesystem import lltype
from pypy.tool.sourcetools import compile2, func_with_new_name
from pypy.rlib.unroll import unrolling_iterable
from pypy.rlib.objectmodel import we_are_translated
from pypy.rlib.nonconst import NonConstant
from pypy.rlib.rarithmetic import r_uint
from pypy.translator.translator import TranslationContext


def is_root(w_obj):
    assert isinstance(w_obj, W_Root)
is_root.expecting = W_Root

def is_arguments(arg):
    assert isinstance(arg, argument.Arguments)
is_arguments.expecting = argument.Arguments


class Entry(ExtRegistryEntry):
    _about_ = is_root, is_arguments

    def compute_result_annotation(self, s_w_obj):
        cls = self.instance.expecting
        s_inst = SomeInstance(self.bookkeeper.getuniqueclassdef(cls),
                              can_be_None=True)
        assert s_inst.contains(s_w_obj)
        return s_None

    def specialize_call(self, hop):
        return hop.inputconst(lltype.Void, None)

# ____________________________________________________________


class FakeObjSpace(ObjSpace):

    def __init__(self):
        self._seen_extras = []
        ObjSpace.__init__(self)

    def str_w(self, w_obj):
        is_root(w_obj)
        return NonConstant("foobar")

    def int_w(self, w_obj):
        is_root(w_obj)
        return NonConstant(-42)

    def float_w(self, w_obj):
        is_root(w_obj)
        return NonConstant(42.5)

    def uint_w(self, w_obj):
        is_root(w_obj)
        return r_uint(NonConstant(42))

    def bigint_w(self, w_obj):
        from pypy.rlib.rbigint import rbigint
        is_root(w_obj)
        return rbigint.fromint(NonConstant(42))

    def unicode_w(self, w_obj):
        is_root(w_obj)
        return NonConstant(u"foobar")

    def is_true(self, w_obj):
        is_root(w_obj)
        return NonConstant(False)

    def unwrap(self, w_obj):
        "NOT_RPYTHON"
        raise NotImplementedError

    def newdict(self, module=False, instance=False, classofinstance=None,
                strdict=False):
        return W_Root()

    def newtuple(self, list_w):
        is_root(list_w[NonConstant(0)])
        return W_Root()

    def newlist(self, list_w):
        is_root(list_w[NonConstant(0)])
        return W_Root()

    def newslice(self, w_start, w_end, w_step):
        is_root(w_start)
        is_root(w_end)
        is_root(w_step)
        return W_Root()

    def marshal_w(self, w_obj):
        "NOT_RPYTHON"
        raise NotImplementedError

    def wrap(self, x):
        if isinstance(x, gateway.interp2app):
            self._see_interp2app(x)
        return W_Root()
    wrap._annspecialcase_ = "specialize:argtype(1)"

    def _see_interp2app(self, interp2app):
        "NOT_RPYTHON"
        activation = interp2app._code.activation
        def check():
            scope_w = [W_Root()] * NonConstant(42)
            w_result = activation._run(self, scope_w)
            is_root(w_result)
        check = func_with_new_name(check, 'check__' + interp2app.name)
        self._seen_extras.append(check)

    def call_args(self, w_func, args):
        is_root(w_func)
        is_arguments(args)
        return W_Root()

    def gettypefor(self, cls):
        return self.gettypeobject(cls.typedef)

    def gettypeobject(self, typedef):
        return self.fromcache(TypeCache).getorbuild(typedef)

    def unpackiterable(self, w_iterable, expected_length=-1):
        is_root(w_iterable)
        if expected_length < 0:
            expected_length = 3
        return [W_Root()] * expected_length

    def allocate_instance(self, cls, w_subtype):
        "NOT_RPYTHON"
        xxx

    # ----------

    def translates(self, func=None, argtypes=None):
        if func is not None:
            if argtypes is None:
                nb_args = func.func_code.co_argcount
                argtypes = [W_Root] * nb_args
        else:
            func = lambda: None
            argtypes = []
        #
        t = TranslationContext()
        self.t = t     # for debugging
        ann = t.buildannotator()
        #
        done = 0
        while True:
            # annotate all _seen_extras, knowing that annotating some may
            # grow the list
            while done < len(self._seen_extras):
                print self._seen_extras
                ann.build_types(self._seen_extras[done], [],
                                complete_now=False)
                done += 1
            # when the list stops growing, really complete
            ann.build_types(func, argtypes, complete_now=True)
            # if the list did not grow because of completion, we are done
            if done == len(self._seen_extras):
                break
        #t.viewcg()
        t.buildrtyper().specialize()
        t.checkgraphs()


def setup():
    for name in (ObjSpace.ConstantTable +
                 ObjSpace.ExceptionTable +
                 ['int', 'str', 'float', 'long', 'tuple', 'list', 'dict']):
        setattr(FakeObjSpace, 'w_' + name, W_Root())
    #
    for (name, _, arity, _) in ObjSpace.MethodTable:
        args = ['w_%d' % i for i in range(arity)]
        d = {'is_root': is_root,
             'W_Root': W_Root}
        exec compile2("""\
            def meth(self, %s):
                %s
                return W_Root()
        """ % (', '.join(args),
               '; '.join(['is_root(%s)' % arg for arg in args]))) in d
        meth = func_with_new_name(d['meth'], name)
        setattr(FakeObjSpace, name, meth)
    #
    for name in ObjSpace.IrregularOpTable:
        assert hasattr(FakeObjSpace, name)    # missing?

setup()

# ____________________________________________________________

class TypeCache(SpaceCache):
    def build(cache, typedef):
        assert isinstance(typedef, TypeDef)
        for value in typedef.rawdict.values():
            cache.space.wrap(value)
        return W_Root()
