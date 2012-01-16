import py
from pypy.objspace.fake.objspace import FakeObjSpace, W_Root
from pypy.interpreter.argument import Arguments
from pypy.interpreter.typedef import TypeDef
from pypy.interpreter.baseobjspace import Wrappable
from pypy.rlib.unroll import unrolling_iterable

def test_create():
    FakeObjSpace()


class TestTranslate:
    def setup_method(self, meth):
        self.space = FakeObjSpace()

    def test_simple(self):
        space = self.space
        space.translates(lambda w_x, w_y: space.add(w_x, w_y))

    def test_methodtable(self):
        space = self.space
        for fixed_arity in [1, 2, 3, 4]:
            #
            methodtable = [name for (name, _, arity, _) in space.MethodTable
                                if arity == fixed_arity]
            methodtable = unrolling_iterable(methodtable)
            args_w = (W_Root(),) * fixed_arity
            #
            def f():
                for name in methodtable:
                    getattr(space, name)(*args_w)
            #
            space.translates(f)

    def test_newdict(self):
        space = self.space
        space.translates(lambda: (space.newdict(),
                                  space.newdict(strdict=True)))

    def test_constants(self):
        space = self.space
        space.translates(lambda: (space.w_None, space.w_True, space.w_False,
                                  space.w_int, space.w_str, space.w_object,
                                  space.w_TypeError))

    def test_wrap(self):
        space = self.space
        space.translates(lambda: (space.wrap(42), space.wrap(42.5),
                                  space.wrap("foo")))

    def test_call_args(self):
        space = self.space
        args = Arguments(space, [W_Root()])
        space.translates(lambda: space.call_args(W_Root(), args))

    def test_gettypefor(self):
        space = self.space
        class W_Foo(Wrappable):
            typedef = TypeDef("foo")
        space.translates(lambda: space.gettypefor(W_Foo))

    def test_is_true(self):
        space = self.space
        space.translates(lambda: space.is_true(W_Root()))
        py.test.raises(AssertionError,
                       space.translates, lambda: space.is_true(42))

    def test_unpackiterable(self):
        space = self.space
        space.translates(lambda: (space.unpackiterable(W_Root()),
                                  space.unpackiterable(W_Root(), 42)))

    def test_newlist(self):
        self.space.newlist([W_Root(), W_Root()])

    def test_default_values(self):
        # the __get__ method takes either 2 or 3 arguments
        space = self.space
        space.translates(lambda: (space.get(W_Root(), W_Root()),
                                  space.get(W_Root(), W_Root(), W_Root())))
