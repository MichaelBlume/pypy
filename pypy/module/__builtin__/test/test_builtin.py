# coding: utf-8
import autopath
import sys
from pypy import conftest

class AppTestBuiltinApp:
    def setup_class(cls):
        class X(object):
            def __eq__(self, other):
                raise OverflowError
            def __hash__(self):
                return 42
        d = {X(): 5}
        try:
            d[X()]
        except OverflowError:
            cls.w_sane_lookup = cls.space.wrap(True)
        except KeyError:
            cls.w_sane_lookup = cls.space.wrap(False)
        # starting with CPython 2.6, when the stack is almost out, we
        # can get a random error, instead of just a RuntimeError.
        # For example if an object x has a __getattr__, we can get
        # AttributeError if attempting to call x.__getattr__ runs out
        # of stack.  That's annoying, so we just work around it.
        if conftest.option.runappdirect:
            cls.w_safe_runtimerror = cls.space.wrap(True)
        else:
            cls.w_safe_runtimerror = cls.space.wrap(sys.version_info < (2, 6))

    def test_bytes_alias(self):
        assert bytes is not str
        assert isinstance(eval("b'hi'"), bytes)

    def test_import(self):
        m = __import__('pprint')
        assert m.pformat({}) == '{}'
        assert m.__name__ == "pprint"
        raises(ImportError, __import__, 'spamspam')
        raises(TypeError, __import__, 1, 2, 3, 4)

    def test_ascii(self):
        assert ascii('') == '\'\''
        assert ascii(0) == '0'
        assert ascii(()) == '()'
        assert ascii([]) == '[]'
        assert ascii({}) == '{}'
        a = []
        a.append(a)
        assert ascii(a) == '[[...]]'
        a = {}
        a[0] = a
        assert ascii(a) == '{0: {...}}'
        # Advanced checks for unicode strings
        def _check_uni(s):
            assert ascii(s) == repr(s)
        _check_uni("'")
        _check_uni('"')
        _check_uni('"\'')
        _check_uni('\0')
        _check_uni('\r\n\t .')
        # Unprintable non-ASCII characters
        _check_uni('\x85')
        _check_uni('\u1fff')
        _check_uni('\U00012fff')
        # Lone surrogates
        _check_uni('\ud800')
        _check_uni('\udfff')
        # Issue #9804: surrogates should be joined even for printable
        # wide characters (UCS-2 builds).
        assert ascii('\U0001d121') == "'\\U0001d121'"
        # another buggy case
        x = ascii("\U00012fff")
        assert x == r"'\U00012fff'"
        # All together
        s = "'\0\"\n\r\t abcd\x85é\U00012fff\uD800\U0001D121xxx."
        assert ascii(s) == \
            r"""'\'\x00"\n\r\t abcd\x85\xe9\U00012fff\ud800\U0001d121xxx.'"""

    def test_bin(self):
        assert bin(0) == "0b0"
        assert bin(-1) == "-0b1"
        assert bin(2) == "0b10"
        assert bin(-2) == "-0b10"
        raises(TypeError, bin, 0.)

    def test_chr(self):
        import sys
        assert chr(65) == 'A'
        assert type(str(65)) is str
        assert chr(0x9876) == '\u9876'
        if sys.maxunicode > 0xFFFF:
            assert chr(sys.maxunicode) == '\U0010FFFF'
            assert chr(0x10000) == '\U00010000'
        else:
            assert chr(sys.maxunicode) == '\uFFFF'
        raises(ValueError, chr, -1)
        raises(ValueError, chr, sys.maxunicode+1)

    def test_globals(self):
        d = {"foo":"bar"}
        exec("def f(): return globals()", d)
        d2 = d["f"]()
        assert d2 is d

    def test_locals(self):
        def f():
            return locals()
        def g(c=0, b=0, a=0):
            return locals()
        assert f() == {}
        assert g() == {'a':0, 'b':0, 'c':0}

    def test_dir(self):
        def f():
            return dir()
        def g(c=0, b=0, a=0):
            return dir()
        def nosp(x): return [y for y in x if y[0]!='_']
        assert f() == []
        assert g() == ['a', 'b', 'c']
        class X(object): pass
        assert nosp(dir(X)) == []
        class X(object):
            a = 23
            c = 45
            b = 67
        assert nosp(dir(X)) == ['a', 'b', 'c']

    def test_dir_in_broken_locals(self):
        class C(object):
            def __getitem__(self, item):
                raise KeyError(item)
            def keys(self):
                return 'abcd'    # not a list!
        names = eval("dir()", {}, C())
        assert names == ['a', 'b', 'c', 'd']

    def test_dir_broken_module(self):
        import types
        class Foo(types.ModuleType):
            __dict__ = 8
        raises(TypeError, dir, Foo("foo"))

    def test_dir_broken_object(self):
        class Foo(object):
            x = 3
            def __getattribute__(self, name):
                return name
        assert dir(Foo()) == []

    def test_dir_custom(self):
        class Foo(object):
            def __dir__(self):
                return ["1", "2", "3"]
        f = Foo()
        assert dir(f) == ["1", "2", "3"]
        class Foo:
            def __dir__(self):
                return ["apple"]
        assert dir(Foo()) == ["apple"]
        class Foo(object):
            def __dir__(self):
                return 42
        f = Foo()
        raises(TypeError, dir, f)
        import types
        class Foo(types.ModuleType):
            def __dir__(self):
                return ["blah"]
        assert dir(Foo("a_mod")) == ["blah"]

    def test_dir_custom_lookup(self):
        class M(type):
            def __dir__(self, *args): return ["14"]
        class X(object):
            __metaclass__ = M
        x = X()
        x.__dir__ = lambda x: ["14"]
        assert dir(x) != ["14"]

    def test_format(self):
        assert format(4) == "4"
        assert format(10, "o") == "12"
        assert format(10, "#o") == "0o12"
        assert format("hi") == "hi"

    def test_vars(self):
        def f():
            return vars()
        def g(c=0, b=0, a=0):
            return vars()
        assert f() == {}
        assert g() == {'a':0, 'b':0, 'c':0}

    def test_getattr(self):
        class a(object):
            i = 5
        assert getattr(a, 'i') == 5
        raises(AttributeError, getattr, a, 'k')
        assert getattr(a, 'k', 42) == 42
        raises(TypeError, getattr, a, b'i')
        raises(TypeError, getattr, a, b'k', 42)

    def test_getattr_typecheck(self):
        class A(object):
            def __getattribute__(self, name):
                pass
            def __setattr__(self, name, value):
                pass
            def __delattr__(self, name):
                pass
        raises(TypeError, getattr, A(), 42)
        raises(TypeError, setattr, A(), 42, 'x')
        raises(TypeError, delattr, A(), 42)

    def test_sum(self):
        assert sum([]) ==0
        assert sum([42]) ==42
        assert sum([1,2,3]) ==6
        assert sum([],5) ==5
        assert sum([1,2,3],4) ==10
        #
        class Foo(object):
            def __radd__(self, other):
                assert other is None
                return 42
        assert sum([Foo()], None) == 42

    def test_type_selftest(self):
        assert type(type) is type

    def test_iter_sequence(self):
        raises(TypeError,iter,3)
        x = iter(['a','b','c'])
        assert next(x) =='a'
        assert next(x) =='b'
        assert next(x) =='c'
        raises(StopIteration, next, x)

    def test_iter___iter__(self):
        # This test assumes that dict.keys() method returns keys in
        # the same order as dict.__iter__().
        # Also, this test is not as explicit as the other tests;
        # it tests 4 calls to __iter__() in one assert.  It could
        # be modified if better granularity on the assert is required.
        mydict = {'a':1,'b':2,'c':3}
        assert list(iter(mydict)) == list(mydict.keys())

    def test_iter_callable_sentinel(self):
        class count(object):
            def __init__(self):
                self.value = 0
            def __call__(self):
                self.value += 1
                return self.value
        # XXX Raising errors is quite slow --
        #            uncomment these lines when fixed
        #self.assertRaises(TypeError,iter,3,5)
        #self.assertRaises(TypeError,iter,[],5)
        #self.assertRaises(TypeError,iter,{},5)
        x = iter(count(),3)
        assert next(x) ==1
        assert next(x) ==2
        raises(StopIteration, next, x)

    def test_enumerate(self):
        seq = range(2,4)
        enum = enumerate(seq)
        assert next(enum) == (0, 2)
        assert next(enum) == (1, 3)
        raises(StopIteration, next, enum)
        raises(TypeError, enumerate, 1)
        raises(TypeError, enumerate, None)
        enum = enumerate(range(5), 2)
        assert list(enum) == zip(range(2, 7), range(5))

    def test_next(self):
        x = iter(['a', 'b', 'c'])
        assert next(x) == 'a'
        assert next(x) == 'b'
        assert next(x) == 'c'
        raises(StopIteration, next, x)
        assert next(x, 42) == 42

    def test_next__next__(self):
        class Counter:
            def __init__(self):
                self.count = 0
            def __next__(self):
                self.count += 1
                return self.count
        x = Counter()
        assert next(x) == 1
        assert next(x) == 2
        assert next(x) == 3

    def test_range_args(self):
##        # range() attributes are deprecated and were removed in Python 2.3.
##        x = range(2)
##        assert x.start == 0
##        assert x.stop == 2
##        assert x.step == 1

##        x = range(2,10,2)
##        assert x.start == 2
##        assert x.stop == 10
##        assert x.step == 2

##        x = range(2.3, 10.5, 2.4)
##        assert x.start == 2
##        assert x.stop == 10
##        assert x.step == 2

        raises(ValueError, range, 0, 1, 0)

    def test_range_repr(self): 
        assert repr(range(1)) == 'range(1)'
        assert repr(range(1,2)) == 'range(1, 2)'
        assert repr(range(1,2,3)) == 'range(1, 2, 3)'

    def test_range_up(self):
        x = range(2)
        iter_x = iter(x)
        assert next(iter_x) == 0
        assert next(iter_x) == 1
        raises(StopIteration, next, iter_x)

    def test_range_down(self):
        x = range(4,2,-1)

        iter_x = iter(x)
        assert next(iter_x) == 4
        assert next(iter_x) == 3
        raises(StopIteration, next, iter_x)

    def test_range_has_type_identity(self):
        assert type(range(1)) == type(range(1))

    def test_range_len(self):
        x = range(33)
        assert len(x) == 33
        raises(TypeError, range, 33.2)
        x = range(33,0,-1)
        assert len(x) == 33
        x = range(33,0)
        assert len(x) == 0
        raises(TypeError, range, 33, 0.2)
        assert len(x) == 0
        x = range(0,33)
        assert len(x) == 33
        x = range(0,33,-1)
        assert len(x) == 0
        x = range(0,33,2)
        assert len(x) == 17
        x = range(0,32,2)
        assert len(x) == 16

    def test_range_indexing(self):
        x = range(0,33,2)
        assert x[7] == 14
        assert x[-7] == 20
        raises(IndexError, x.__getitem__, 17)
        raises(IndexError, x.__getitem__, -18)
        assert list(x.__getitem__(slice(0,3,1))) == [0, 2, 4]

    def test_range_bad_args(self):
        raises(TypeError, range, '1')
        raises(TypeError, range, None)
        raises(TypeError, range, 3+2j)
        raises(TypeError, range, 1, '1')
        raises(TypeError, range, 1, 3+2j)
        raises(TypeError, range, 1, 2, '1')
        raises(TypeError, range, 1, 2, 3+2j)
    
    def test_sorted(self):
        l = []
        sorted_l = sorted(l)
        assert sorted_l is not l
        assert sorted_l == l
        l = [1, 5, 2, 3]
        sorted_l = sorted(l)
        assert sorted_l == [1, 2, 3, 5]

    def test_sorted_with_keywords(self):
        l = ['a', 'C', 'b']
        sorted_l = sorted(l, reverse = True)
        assert sorted_l is not l
        assert sorted_l == ['b', 'a', 'C']
        sorted_l = sorted(l, reverse = True, key = lambda x: x.lower())
        assert sorted_l is not l
        assert sorted_l == ['C', 'b', 'a']
        
    def test_reversed_simple_sequences(self):
        l = range(5)
        rev = reversed(l)
        assert list(rev) == [4, 3, 2, 1, 0]
        assert list(l.__reversed__()) == [4, 3, 2, 1, 0]
        s = "abcd"
        assert list(reversed(s)) == ['d', 'c', 'b', 'a']

    def test_reversed_custom_objects(self):
        """make sure __reversed__ is called when defined"""
        class SomeClass(object):
            def __reversed__(self):
                return 42
        obj = SomeClass()
        assert reversed(obj) == 42
    
        
    def test_cmp(self):
        assert cmp(9,9) == 0
        assert cmp(0,9) < 0
        assert cmp(9,0) > 0
        assert cmp(b"abc", 12) != 0
        assert cmp("abc", 12) != 0

    def test_cmp_more(self):
        class C(object):
            def __eq__(self, other):
                return True
            def __cmp__(self, other):
                raise RuntimeError
        c1 = C()
        c2 = C()
        raises(RuntimeError, cmp, c1, c2)

    def test_cmp_cyclic(self):
        if not self.sane_lookup:
            skip("underlying Python implementation has insane dict lookup")
        if not self.safe_runtimerror:
            skip("underlying Python may raise random exceptions on stack ovf")
        a = []; a.append(a)
        b = []; b.append(b)
        from UserList import UserList
        c = UserList(); c.append(c)
        raises(RuntimeError, cmp, a, b)
        raises(RuntimeError, cmp, b, c)
        raises(RuntimeError, cmp, c, a)
        raises(RuntimeError, cmp, a, c)
        # okay, now break the cycles
        a.pop(); b.pop(); c.pop()

    def test_return_None(self):
        class X(object): pass
        x = X()
        assert setattr(x, 'x', 11) == None
        assert delattr(x, 'x') == None
        # To make this test, we need autopath to work in application space.
        #self.assertEquals(execfile('emptyfile.py'), None)

    def test_divmod(self):
        assert divmod(15,10) ==(1,5)

    def test_callable(self):
        class Call(object):
            def __call__(self, a):
                return a+2
        assert callable(Call()), (
                    "Builtin function 'callable' misreads callable object")
        assert callable(int), (
                    "Builtin function 'callable' misreads int")
        class Call:
            def __call__(self, a):
                return a+2
        assert callable(Call())


    def test_uncallable(self):
        # XXX TODO: I made the NoCall class explicitly newstyle to try and
        # remedy the failure in this test observed when running this with
        # the trivial objectspace, but the test _still_ fails then (it
        # doesn't fail with the standard objectspace, though).
        class NoCall(object):
            pass
        a = NoCall()
        assert not callable(a), (
                    "Builtin function 'callable' misreads uncallable object")
        a.__call__ = lambda: "foo"
        assert not callable(a), (
                    "Builtin function 'callable' tricked by instance-__call__")
        class NoCall:
            pass
        assert not callable(NoCall())

    def test_hash(self):
        assert hash(23) == hash(23)
        assert hash(2.3) == hash(2.3)
        assert hash('23') == hash("23")
        assert hash((23,)) == hash((23,))
        assert hash(22) != hash(23)
        raises(TypeError, hash, [])
        raises(TypeError, hash, {})

    def test_eval(self):
        assert eval("1+2") == 3
        assert eval(" \t1+2\n") == 3
        assert eval("len([])") == 0
        assert eval("len([])", {}) == 0        
        # cpython 2.4 allows this (raises in 2.3)
        assert eval("3", None, None) == 3
        i = 4
        assert eval("i", None, None) == 4
        assert eval('a', None, dict(a=42)) == 42

    def test_compile(self):
        co = compile('1+2', '?', 'eval')
        assert eval(co) == 3
        compile("from __future__ import with_statement", "<test>", "exec")
        raises(SyntaxError, compile, '-', '?', 'eval')
        raises(SyntaxError, compile, '"\\xt"', '?', 'eval')
        raises(ValueError, compile, '1+2', '?', 'maybenot')
        raises(ValueError, compile, "\n", "<string>", "exec", 0xff)
        raises(TypeError, compile, '1+2', 12, 34)

    def test_unicode_encoding_compile(self):
        code = "# -*- coding: utf-8 -*-\npass\n"
        raises(SyntaxError, compile, code, "tmp", "exec")

    def test_bytes_compile(self):
        code = b"# -*- coding: utf-8 -*-\npass\n"
        compile(code, "tmp", "exec")

    def test_recompile_ast(self):
        import _ast
        # raise exception when node type doesn't match with compile mode
        co1 = compile('print(1)', '<string>', 'exec', _ast.PyCF_ONLY_AST)
        raises(TypeError, compile, co1, '<ast>', 'eval')
        co2 = compile('1+1', '<string>', 'eval', _ast.PyCF_ONLY_AST)
        compile(co2, '<ast>', 'eval')

    def test_isinstance(self):
        assert isinstance(5, int)
        assert isinstance(5, object)
        assert not isinstance(5, float)
        assert isinstance(True, (int, float))
        assert not isinstance(True, (type, float))
        assert isinstance(True, ((type, float), bool))
        raises(TypeError, isinstance, 5, 6)
        raises(TypeError, isinstance, 5, (float, 6))

    def test_issubclass(self):
        assert issubclass(int, int)
        assert issubclass(int, object)
        assert not issubclass(int, float)
        assert issubclass(bool, (int, float))
        assert not issubclass(bool, (type, float))
        assert issubclass(bool, ((type, float), bool))
        raises(TypeError, issubclass, 5, int)
        raises(TypeError, issubclass, int, 6)
        raises(TypeError, issubclass, int, (float, 6))

    def test_staticmethod(self):
        class X(object):
            def f(*args, **kwds): return args, kwds
            f = staticmethod(f)
        assert X.f() == ((), {})
        assert X.f(42, x=43) == ((42,), {'x': 43})
        assert X().f() == ((), {})
        assert X().f(42, x=43) == ((42,), {'x': 43})

    def test_classmethod(self):
        class X(object):
            def f(*args, **kwds): return args, kwds
            f = classmethod(f)
        class Y(X):
            pass
        assert X.f() == ((X,), {})
        assert X.f(42, x=43) == ((X, 42), {'x': 43})
        assert X().f() == ((X,), {})
        assert X().f(42, x=43) == ((X, 42), {'x': 43})
        assert Y.f() == ((Y,), {})
        assert Y.f(42, x=43) == ((Y, 42), {'x': 43})
        assert Y().f() == ((Y,), {})
        assert Y().f(42, x=43) == ((Y, 42), {'x': 43})

    def test_hasattr(self):
        class X(object):
            def broken(): pass   # TypeError
            abc = property(broken)
            def broken2(): raise IOError
            bac = property(broken2)
        x = X()
        x.foo = 42
        assert hasattr(x, '__class__') is True
        assert hasattr(x, 'foo') is True
        assert hasattr(x, 'bar') is False
        raises(TypeError, "hasattr(x, 'abc')")
        raises(TypeError, "hasattr(x, 'bac')")
        raises(TypeError, hasattr, x, None)
        raises(TypeError, hasattr, x, 42)
        assert hasattr(x, '\u5678') is False

    def test_hasattr_exception(self):
        class X(object):
            def __getattr__(self, name):
                if name == 'foo':
                    raise AttributeError
                else:
                    raise KeyError
        x = X()
        assert hasattr(x, 'foo') is False
        raises(KeyError, "hasattr(x, 'bar')")

    def test_compile_leading_newlines(self):
        src = """
def fn(): pass
"""
        co = compile(src, 'mymod', 'exec')
        firstlineno = co.co_firstlineno
        assert firstlineno == 2

    def test_print_function(self):
        import builtins
        import sys
        import io
        pr = getattr(builtins, "print")
        save = sys.stdout
        out = sys.stdout = io.StringIO()
        try:
            pr("Hello,", "person!")
        finally:
            sys.stdout = save
        assert out.getvalue() == "Hello, person!\n"
        out = io.StringIO()
        pr("Hello,", "person!", file=out)
        assert out.getvalue() == "Hello, person!\n"
        out = io.StringIO()
        pr("Hello,", "person!", file=out, end="")
        assert out.getvalue() == "Hello, person!"
        out = io.StringIO()
        pr("Hello,", "person!", file=out, sep="X")
        assert out.getvalue() == "Hello,Xperson!\n"
        out = io.StringIO()
        pr(b"Hello,", b"person!", file=out)
        result = out.getvalue()
        assert isinstance(result, str)
        print('XXXXXX', result)
        assert result == "b'Hello,' b'person!'\n"
        pr("Hello", file=None) # This works.
        out = io.StringIO()
        pr(None, file=out)
        assert out.getvalue() == "None\n"

    def test_print_exceptions(self):
        import builtins
        pr = getattr(builtins, "print")
        raises(TypeError, pr, x=3)
        raises(TypeError, pr, end=3)
        raises(TypeError, pr, sep=42)

    def test_round(self):
        assert round(11.234) == 11.0
        assert round(11.234, -1) == 10.0
        assert round(11.234, 0) == 11.0
        assert round(11.234, 1) == 11.2
        #
        assert round(5e15-1) == 5e15-1
        assert round(5e15) == 5e15
        assert round(-(5e15-1)) == -(5e15-1)
        assert round(-5e15) == -5e15
        #
        inf = 1e200 * 1e200
        assert round(inf) == inf
        assert round(-inf) == -inf
        nan = inf / inf
        assert repr(round(nan)) == repr(nan)
        #
        raises(OverflowError, round, 1.6e308, -308)
        #
        assert round(562949953421312.5, 1) == 562949953421312.5
        assert round(56294995342131.5, 3) == 56294995342131.5

    def test_vars_obscure_case(self):
        class C_get_vars(object):
            def getDict(self):
                return {'a':2}
            __dict__ = property(fget=getDict)
        assert vars(C_get_vars()) == {'a':2}
