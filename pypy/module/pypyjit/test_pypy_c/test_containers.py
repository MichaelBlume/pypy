
import py, sys
from pypy.module.pypyjit.test_pypy_c.test_00_model import BaseTestPyPyC


class TestDicts(BaseTestPyPyC):
    def test_strdict(self):
        def fn(n):
            import sys
            d = {}
            class A(object):
                pass
            a = A()
            a.x = 1
            for s in sys.modules.keys() * 1000:
                inc = a.x # ID: look
                d[s] = d.get(s, 0) + inc
            return sum(d.values())
        #
        log = self.run(fn, [1000])
        assert log.result % 1000 == 0
        loop, = log.loops_by_filename(self.filepath)
        ops = loop.ops_by_id('look')
        assert log.opnames(ops) == ['setfield_gc',
                                    'guard_not_invalidated']

    def test_identitydict(self):
        def fn(n):
            class X(object):
                pass
            x = X()
            d = {}
            d[x] = 1
            res = 0
            for i in range(300):
                value = d[x]  # ID: getitem
                res += value
            return res
        #
        log = self.run(fn, [1000])
        assert log.result == 300
        loop, = log.loops_by_filename(self.filepath)
        # check that the call to ll_dict_lookup is not a call_may_force, the
        # gc_id call is hoisted out of the loop, the id of a value obviously
        # can't change ;)
        assert loop.match_by_id("getitem", """
            i28 = call(ConstClass(ll_dict_lookup__dicttablePtr_objectPtr_Signed), p18, p6, i25, descr=...)
            ...
            p33 = call(ConstClass(ll_get_value__dicttablePtr_Signed), p18, i28, descr=...)
            ...
        """)
