from pypy.module.pypyjit.test_pypy_c.test_00_model import BaseTestPyPyC


class TestGenerators(BaseTestPyPyC):
    def test_simple_generator(self):
        def main(n):
            def f():
                for i in range(10000):
                    i -= 1
                    i -= 42    # ID: subtract
                    yield i

            def g():
                for i in f():  # ID: generator
                    pass

            g()

        log = self.run(main, [500])
        # XXX XXX this test fails so far because of a detail that
        # changed with jit-simplify-backendintf.  We should try to
        # think of a way to be more resistent against such details.
        # The issue is that we now get one Tracing, then go back
        # to the interpreter hoping to immediately run the JITted
        # code; but instead, we Trace again, just because another
        # counter was also about to reach its limit...
        loop, = log.loops_by_filename(self.filepath)
        assert loop.match_by_id("generator", """
            i16 = force_token()
            p45 = new_with_vtable(ConstClass(W_IntObject))
            setfield_gc(p45, i29, descr=<SignedFieldDescr .*>)
            setarrayitem_gc(p8, 0, p45, descr=<GcPtrArrayDescr>)
            i47 = arraylen_gc(p8, descr=<GcPtrArrayDescr>) # Should be removed by backend
            jump(..., descr=...)
            """)
        assert loop.match_by_id("subtract", """
            setfield_gc(p7, 35, descr=<.*last_instr .*>)      # XXX bad, kill me
            i2 = int_sub_ovf(i1, 42)
            guard_no_overflow(descr=...)
            """)
