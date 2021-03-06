import gc
from pypy.module.thread.ll_thread import *
from pypy.translator.c.test.test_boehm import AbstractGCTestClass
from pypy.rpython.lltypesystem import lltype, rffi
import py

def setup_module(mod):
    # Hack to avoid a deadlock if the module is run after other test files :-(
    # In this module, we assume that ll_thread.start_new_thread() is not
    # providing us with a GIL equivalent, except in test_gc_locking
    # which installs its own aroundstate.
    rffi.aroundstate._freeze_()

def test_lock():
    l = allocate_lock()
    ok1 = l.acquire(True)
    ok2 = l.acquire(False)
    l.release()
    ok3 = l.acquire(False)
    res = ok1 and not ok2 and ok3
    assert res == 1

def test_thread_error():
    l = allocate_lock()
    try:
        l.release()
    except error:
        pass
    else:
        py.test.fail("Did not raise")


class AbstractThreadTests(AbstractGCTestClass):
    use_threads = True

    def test_start_new_thread(self):
        import time

        class State:
            pass
        state = State()

        def bootstrap1():
            state.my_thread_ident1 = get_ident()
        def bootstrap2():
            state.my_thread_ident2 = get_ident()

        def f():
            state.my_thread_ident1 = get_ident()
            state.my_thread_ident2 = get_ident()
            start_new_thread(bootstrap1, ())
            start_new_thread(bootstrap2, ())
            willing_to_wait_more = 1000
            while (state.my_thread_ident1 == get_ident() or
                   state.my_thread_ident2 == get_ident()):
                willing_to_wait_more -= 1
                if not willing_to_wait_more:
                    raise Exception("thread didn't start?")
                time.sleep(0.01)
            return 42

        fn = self.getcompiled(f, [])
        res = fn()
        assert res == 42

    def test_gc_locking(self):
        import time
        from pypy.rlib.objectmodel import invoke_around_extcall
        from pypy.rlib.objectmodel import we_are_translated
        from pypy.rlib.debug import ll_assert

        class State:
            pass
        state = State()

        class Z:
            def __init__(self, i, j):
                self.i = i
                self.j = j
            def run(self):
                j = self.j
                if self.i > 1:
                    g(self.i-1, self.j * 2)
                    ll_assert(j == self.j, "1: bad j")
                    g(self.i-2, self.j * 2 + 1)
                else:
                    if len(state.answers) % 7 == 5:
                        gc.collect()
                    state.answers.append(self.j)
                ll_assert(j == self.j, "2: bad j")
            run._dont_inline_ = True

        def before_extcall():
            release_NOAUTO(state.gil)
        before_extcall._gctransformer_hint_cannot_collect_ = True
        # ^^^ see comments in gil.py about this hint

        def after_extcall():
            acquire_NOAUTO(state.gil, True)
            gc_thread_run()
        after_extcall._gctransformer_hint_cannot_collect_ = True
        # ^^^ see comments in gil.py about this hint

        def bootstrap():
            # after_extcall() is called before we arrive here.
            # We can't just acquire and release the GIL manually here,
            # because it is unsafe: bootstrap() is called from a rffi
            # callback which checks for and reports exceptions after
            # bootstrap() returns.  The exception checking code must be
            # protected by the GIL too.
            z = state.z
            state.z = None
            state.bootstrapping.release()
            z.run()
            gc_thread_die()
            # before_extcall() is called after we leave here

        def g(i, j):
            state.bootstrapping.acquire(True)
            state.z = Z(i, j)
            gc_thread_prepare()
            start_new_thread(bootstrap, ())

        def f():
            state.gil = allocate_ll_lock()
            acquire_NOAUTO(state.gil, True)
            state.bootstrapping = allocate_lock()
            state.answers = []
            state.finished = 0
            # the next line installs before_extcall() and after_extcall()
            # to be called automatically around external function calls.
            # When not translated it does not work around time.sleep(),
            # so we have to call them manually for this test.
            invoke_around_extcall(before_extcall, after_extcall)

            g(10, 1)
            done = False
            willing_to_wait_more = 2000
            while not done:
                if not willing_to_wait_more:
                    break
                willing_to_wait_more -= 1
                done = len(state.answers) == expected

                if not we_are_translated(): before_extcall()
                time.sleep(0.01)
                if not we_are_translated(): after_extcall()

            if not we_are_translated(): before_extcall()
            time.sleep(0.1)
            if not we_are_translated(): after_extcall()

            return len(state.answers)

        expected = 89
        try:
            fn = self.getcompiled(f, [])
        finally:
            rffi.aroundstate._freeze_()
        answers = fn()
        assert answers == expected

class TestRunDirectly(AbstractThreadTests):
    def getcompiled(self, f, argtypes):
        return f

    def test_start_new_thread(self):
        py.test.skip("deadlocks occasionally -- why???")

class TestUsingBoehm(AbstractThreadTests):
    gcpolicy = 'boehm'

class TestUsingFramework(AbstractThreadTests):
    gcpolicy = 'generation'
