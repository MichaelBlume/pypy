from pypy.conftest import gettestobjspace


class AppTestClasses:
    def setup_class(cls):
        cls.space = gettestobjspace(usemodules=['_multibytecodec'])
        cls.w_IncrementalHzDecoder = cls.space.appexec([], """():
            import _codecs_cn
            from _multibytecodec import MultibyteIncrementalDecoder

            class IncrementalHzDecoder(MultibyteIncrementalDecoder):
                codec = _codecs_cn.getcodec('hz')

            return IncrementalHzDecoder
        """)

    def test_decode_hz(self):
        d = self.IncrementalHzDecoder()
        r = d.decode("~{abcd~}")
        assert r == u'\u5f95\u6c85'
        r = d.decode("~{efgh~}")
        assert r == u'\u5f50\u73b7'
        for c, output in zip("!~{abcd~}xyz~{efgh",
              [u'!',  # !
               u'',   # ~
               u'',   # {
               u'',   # a
               u'\u5f95',   # b
               u'',   # c
               u'\u6c85',   # d
               u'',   # ~
               u'',   # }
               u'x',  # x
               u'y',  # y
               u'z',  # z
               u'',   # ~
               u'',   # {
               u'',   # e
               u'\u5f50',   # f
               u'',   # g
               u'\u73b7',   # h
               ]):
            r = d.decode(c)
            assert r == output

    def test_decode_hz_final(self):
        d = self.IncrementalHzDecoder()
        r = d.decode("~{", True)
        assert r == u''
        raises(UnicodeDecodeError, d.decode, "~", True)
        raises(UnicodeDecodeError, d.decode, "~{a", True)

    def test_decode_hz_reset(self):
        d = self.IncrementalHzDecoder()
        r = d.decode("ab")
        assert r == u'ab'
        r = d.decode("~{")
        assert r == u''
        r = d.decode("ab")
        assert r == u'\u5f95'
        r = d.decode("ab")
        assert r == u'\u5f95'
        d.reset()
        r = d.decode("ab")
        assert r == u'ab'
