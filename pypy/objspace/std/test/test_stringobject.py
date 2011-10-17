from pypy.objspace.std import stringobject
from pypy.objspace.std.stringobject import W_StringObject
from pypy.conftest import gettestobjspace


class TestW_StringObject:

    def teardown_method(self, method):
        pass

    def test_bytes_w(self):
        assert self.space.bytes_w(self.space.wrapbytes("foo")) == "foo"

    def test_equality(self):
        w = self.space.wrapbytes
        assert self.space.eq_w(w('abc'), w('abc'))
        assert not self.space.eq_w(w('abc'), w('def'))

    def test_order_cmp(self):
        space = self.space
        w = space.wrapbytes
        assert self.space.is_true(space.lt(w('a'), w('b')))
        assert self.space.is_true(space.lt(w('a'), w('ab')))
        assert self.space.is_true(space.le(w('a'), w('a')))
        assert self.space.is_true(space.gt(w('a'), w('')))

    def test_truth(self):
        w = self.space.wrapbytes
        assert self.space.is_true(w('non-empty'))
        assert not self.space.is_true(w(''))

    def test_getitem(self):
        space = self.space
        w = space.wrap
        w_str = space.wrapbytes('abc')
        assert space.eq_w(space.getitem(w_str, w(0)), w(ord('a')))
        assert space.eq_w(space.getitem(w_str, w(-1)), w(ord('c')))
        self.space.raises_w(space.w_IndexError,
                            space.getitem,
                            w_str,
                            w(3))

    def test_slice(self):
        space = self.space
        w = space.wrap
        wb = space.wrapbytes
        w_str = wb('abc')

        w_slice = space.newslice(w(0), w(0), space.w_None)
        assert self.space.eq_w(space.getitem(w_str, w_slice), wb(''))

        w_slice = space.newslice(w(0), w(1), space.w_None)
        assert self.space.eq_w(space.getitem(w_str, w_slice), wb('a'))

        w_slice = space.newslice(w(0), w(10), space.w_None)
        assert self.space.eq_w(space.getitem(w_str, w_slice), wb('abc'))

        w_slice = space.newslice(space.w_None, space.w_None, space.w_None)
        assert self.space.eq_w(space.getitem(w_str, w_slice), wb('abc'))

        w_slice = space.newslice(space.w_None, w(-1), space.w_None)
        assert self.space.eq_w(space.getitem(w_str, w_slice), wb('ab'))

        w_slice = space.newslice(w(-1), space.w_None, space.w_None)
        assert self.space.eq_w(space.getitem(w_str, w_slice), wb('c'))

    def test_extended_slice(self):
        space = self.space
        if self.space.__class__.__name__.startswith('Trivial'):
            import sys
            if sys.version < (2, 3):
                return
        w_None = space.w_None
        w = space.wrap
        wb = space.wrapbytes
        w_str = wb('hello')

        w_slice = space.newslice(w_None, w_None, w(1))
        assert self.space.eq_w(space.getitem(w_str, w_slice), wb('hello'))

        w_slice = space.newslice(w_None, w_None, w(-1))
        assert self.space.eq_w(space.getitem(w_str, w_slice), wb('olleh'))

        w_slice = space.newslice(w_None, w_None, w(2))
        assert self.space.eq_w(space.getitem(w_str, w_slice), wb('hlo'))

        w_slice = space.newslice(w(1), w_None, w(2))
        assert self.space.eq_w(space.getitem(w_str, w_slice), wb('el'))

class AppTestStringObject:

    def test_format(self):
        import operator
        raises(TypeError, operator.mod, b"%s", (1,))

    def test_split(self):
        assert b"".split() == []
        assert b"".split(b'x') == [b'']
        assert b" ".split() == []
        assert b"a".split() == [b'a']
        assert b"a".split(b"a", 1) == [b'', b'']
        assert b" ".split(b" ", 1) == [b'', b'']
        assert b"aa".split(b"a", 2) == [b'', b'', b'']
        assert b" a ".split() == [b'a']
        assert b"a b c".split() == [b'a',b'b',b'c']
        assert b'this is the split function'.split() == [
            b'this', b'is', b'the', b'split', b'function']
        assert b'a|b|c|d'.split(b'|') == [b'a', b'b', b'c', b'd']
        assert b'a|b|c|d'.split(b'|', 2) == [b'a', b'b', b'c|d']
        assert b'a b c d'.split(None, 1) == [b'a', b'b c d']
        assert b'a b c d'.split(None, 2) == [b'a', b'b', b'c d']
        assert b'a b c d'.split(None, 3) == [b'a', b'b', b'c', b'd']
        assert b'a b c d'.split(None, 4) == [b'a', b'b', b'c', b'd']
        assert b'a b c d'.split(None, 0) == [b'a b c d']
        assert b'a  b  c  d'.split(None, 2) == [b'a', b'b', b'c  d']
        assert b'a b c d '.split() == [b'a', b'b', b'c', b'd']
        assert b'a//b//c//d'.split(b'//') == [b'a', b'b', b'c', b'd']
        assert b'endcase test'.split(b'test') == [b'endcase ', b'']
        raises(ValueError, b'abc'.split, b'')

    def test_rsplit(self):
        assert b"".rsplit() == []
        assert b" ".rsplit() == []
        assert b"a".rsplit() == [b'a']
        assert b"a".rsplit(b"a", 1) == [b'', b'']
        assert b" ".rsplit(b" ", 1) == [b'', b'']
        assert b"aa".rsplit(b"a", 2) == [b'', b'', b'']
        assert b" a ".rsplit() == [b'a']
        assert b"a b c".rsplit() == [b'a',b'b',b'c']
        assert b'this is the rsplit function'.rsplit() == [
            b'this', b'is', b'the', b'rsplit', b'function']
        assert b'a|b|c|d'.rsplit(b'|') == [b'a', b'b', b'c', b'd']
        assert b'a|b|c|d'.rsplit(b'|', 2) == [b'a|b', b'c', b'd']
        assert b'a b c d'.rsplit(None, 1) == [b'a b c', b'd']
        assert b'a b c d'.rsplit(None, 2) == [b'a b', b'c', b'd']
        assert b'a b c d'.rsplit(None, 3) == [b'a', b'b', b'c', b'd']
        assert b'a b c d'.rsplit(None, 4) == [b'a', b'b', b'c', b'd']
        assert b'a b c d'.rsplit(None, 0) == [b'a b c d']
        assert b'a  b  c  d'.rsplit(None, 2) == [b'a  b', b'c', b'd']
        assert b'a b c d '.rsplit() == [b'a', b'b', b'c', b'd']
        assert b'a//b//c//d'.rsplit(b'//') == [b'a', b'b', b'c', b'd']
        assert b'endcase test'.rsplit(b'test') == [b'endcase ', b'']
        raises(ValueError, b'abc'.rsplit, b'')

    def test_split_splitchar(self):
        assert b"/a/b/c".split(b'/') == [b'',b'a',b'b',b'c']

    def test_title(self):
        assert b"brown fox".title() == b"Brown Fox"
        assert b"!brown fox".title() == b"!Brown Fox"
        assert b"bROWN fOX".title() == b"Brown Fox"
        assert b"Brown Fox".title() == b"Brown Fox"
        assert b"bro!wn fox".title() == b"Bro!Wn Fox"

    def test_istitle(self):
        assert b"".istitle() == False
        assert b"!".istitle() == False
        assert b"!!".istitle() == False
        assert b"brown fox".istitle() == False
        assert b"!brown fox".istitle() == False
        assert b"bROWN fOX".istitle() == False
        assert b"Brown Fox".istitle() == True
        assert b"bro!wn fox".istitle() == False
        assert b"Bro!wn fox".istitle() == False
        assert b"!brown Fox".istitle() == False
        assert b"!Brown Fox".istitle() == True
        assert b"Brow&&&&N Fox".istitle() == True
        assert b"!Brow&&&&n Fox".istitle() == False

    def test_capitalize(self):
        assert b"brown fox".capitalize() == b"Brown fox"
        assert b' hello '.capitalize() == b' hello '
        assert b'Hello '.capitalize() == b'Hello '
        assert b'hello '.capitalize() == b'Hello '
        assert b'aaaa'.capitalize() == b'Aaaa'
        assert b'AaAa'.capitalize() == b'Aaaa'

    def test_rjust(self):
        s = b"abc"
        assert s.rjust(2) == s
        assert s.rjust(3) == s
        assert s.rjust(4) == b" " + s
        assert s.rjust(5) == b"  " + s
        assert b'abc'.rjust(10) == b'       abc'
        assert b'abc'.rjust(6) == b'   abc'
        assert b'abc'.rjust(3) == b'abc'
        assert b'abc'.rjust(2) == b'abc'
        assert b'abc'.rjust(5, '*') == b'**abc'     # Python 2.4
        raises(TypeError, b'abc'.rjust, 5, 'xx')

    def test_ljust(self):
        s = b"abc"
        assert s.ljust(2) == s
        assert s.ljust(3) == s
        assert s.ljust(4) == s + b" "
        assert s.ljust(5) == s + b"  "
        assert b'abc'.ljust(10) == b'abc       '
        assert b'abc'.ljust(6) == b'abc   '
        assert b'abc'.ljust(3) == b'abc'
        assert b'abc'.ljust(2) == b'abc'
        assert b'abc'.ljust(5, '*') == b'abc**'     # Python 2.4
        raises(TypeError, b'abc'.ljust, 6, '')

    def test_replace(self):
        assert b'one!two!three!'.replace(b'!', b'@', 1) == b'one@two!three!'
        assert b'one!two!three!'.replace(b'!', b'') == b'onetwothree'
        assert b'one!two!three!'.replace(b'!', b'@', 2) == b'one@two@three!'
        assert b'one!two!three!'.replace(b'!', b'@', 3) == b'one@two@three@'
        assert b'one!two!three!'.replace(b'!', b'@', 4) == b'one@two@three@'
        assert b'one!two!three!'.replace(b'!', b'@', 0) == b'one!two!three!'
        assert b'one!two!three!'.replace(b'!', b'@') == b'one@two@three@'
        assert b'one!two!three!'.replace(b'x', b'@') == b'one!two!three!'
        assert b'one!two!three!'.replace(b'x', b'@', 2) == b'one!two!three!'
        assert b'abc'.replace(b'', b'-') == b'-a-b-c-'
        assert b'abc'.replace(b'', b'-', 3) == b'-a-b-c'
        assert b'abc'.replace(b'', b'-', 0) == b'abc'
        assert b''.replace(b'', b'') == b''
        assert b''.replace(b'', b'a') == b'a'
        assert b'abc'.replace(b'ab', b'--', 0) == b'abc'
        assert b'abc'.replace(b'xy', b'--') == b'abc'
        assert b'123'.replace(b'123', b'') == b''
        assert b'123123'.replace(b'123', b'') == b''
        assert b'123x123'.replace(b'123', b'') == b'x'

    def test_replace_buffer(self):
        assert b'one'.replace(buffer(b'o'), buffer(b'n'), 1) == b'nne'
        assert b'one'.replace(buffer(b'o'), buffer(b'n')) == b'nne'

    def test_strip(self):
        s = " a b "
        assert s.strip() == "a b"
        assert s.rstrip() == " a b"
        assert s.lstrip() == "a b "
        assert b'xyzzyhelloxyzzy'.strip(b'xyz') == b'hello'
        assert b'xyzzyhelloxyzzy'.lstrip(b'xyz') == b'helloxyzzy'
        assert b'xyzzyhelloxyzzy'.rstrip(b'xyz') == b'xyzzyhello'

    def test_zfill(self):
        assert b'123'.zfill(2) == b'123'
        assert b'123'.zfill(3) == b'123'
        assert b'123'.zfill(4) == b'0123'
        assert b'+123'.zfill(3) == b'+123'
        assert b'+123'.zfill(4) == b'+123'
        assert b'+123'.zfill(5) == b'+0123'
        assert b'-123'.zfill(3) == b'-123'
        assert b'-123'.zfill(4) == b'-123'
        assert b'-123'.zfill(5) == b'-0123'
        assert b''.zfill(3) == b'000'
        assert b'34'.zfill(1) == b'34'
        assert b'34'.zfill(4) == b'0034'

    def test_center(self):
        s="a b"
        assert s.center(0) == "a b"
        assert s.center(1) == "a b"
        assert s.center(2) == "a b"
        assert s.center(3) == "a b"
        assert s.center(4) == "a b "
        assert s.center(5) == " a b "
        assert s.center(6) == " a b  "
        assert s.center(7) == "  a b  "
        assert s.center(8) == "  a b   "
        assert s.center(9) == "   a b   "
        assert b'abc'.center(10) == b'   abc    '
        assert b'abc'.center(6) == b' abc  '
        assert b'abc'.center(3) == b'abc'
        assert b'abc'.center(2) == b'abc'
        assert b'abc'.center(5, '*') == b'*abc*'     # Python 2.4
        raises(TypeError, b'abc'.center, 4, 'cba')
        assert b' abc'.center(7) == b'   abc '

    def test_count(self):
        assert b"".count(b"x") ==0
        assert b"".count(b"") ==1
        assert b"Python".count(b"") ==7
        assert b"ab aaba".count(b"ab") ==2
        assert b'aaa'.count(b'a') == 3
        assert b'aaa'.count(b'b') == 0
        assert b'aaa'.count(b'a', -1) == 1
        assert b'aaa'.count(b'a', -10) == 3
        assert b'aaa'.count(b'a', 0, -1) == 2
        assert b'aaa'.count(b'a', 0, -10) == 0
        assert b'ababa'.count(b'aba') == 1

    def test_startswith(self):
        assert b'ab'.startswith(b'ab') is True
        assert b'ab'.startswith(b'a') is True
        assert b'ab'.startswith(b'') is True
        assert b'x'.startswith(b'a') is False
        assert b'x'.startswith(b'x') is True
        assert b''.startswith(b'') is True
        assert b''.startswith(b'a') is False
        assert b'x'.startswith(b'xx') is False
        assert b'y'.startswith(b'xx') is False

    def test_startswith_more(self):
        assert b'ab'.startswith(b'a', 0) is True
        assert b'ab'.startswith(b'a', 1) is False
        assert b'ab'.startswith(b'b', 1) is True
        assert b'abc'.startswith(b'bc', 1, 2) is False
        assert b'abc'.startswith(b'c', -1, 4) is True

    def test_startswith_tuples(self):
        assert b'hello'.startswith((b'he', b'ha'))
        assert not b'hello'.startswith((b'lo', b'llo'))
        assert b'hello'.startswith((b'hellox', b'hello'))
        assert not b'hello'.startswith(())
        assert b'helloworld'.startswith((b'hellowo', b'rld', b'lowo'), 3)
        assert not b'helloworld'.startswith((b'hellowo', b'ello', b'rld'), 3)
        assert b'hello'.startswith((b'lo', b'he'), 0, -1)
        assert not b'hello'.startswith((b'he', b'hel'), 0, 1)
        assert b'hello'.startswith((b'he', b'hel'), 0, 2)
        raises(TypeError, b'hello'.startswith, (42,))

    def test_endswith(self):
        assert b'ab'.endswith(b'ab') is True
        assert b'ab'.endswith(b'b') is True
        assert b'ab'.endswith(b'') is True
        assert b'x'.endswith(b'a') is False
        assert b'x'.endswith(b'x') is True
        assert b''.endswith(b'') is True
        assert b''.endswith(b'a') is False
        assert b'x'.endswith(b'xx') is False
        assert b'y'.endswith(b'xx') is False

    def test_endswith_more(self):
        assert b'abc'.endswith(b'ab', 0, 2) is True
        assert b'abc'.endswith(b'bc', 1) is True
        assert b'abc'.endswith(b'bc', 2) is False
        assert b'abc'.endswith(b'b', -3, -1) is True

    def test_endswith_tuple(self):
        assert not b'hello'.endswith((b'he', b'ha'))
        assert b'hello'.endswith((b'lo', b'llo'))
        assert b'hello'.endswith((b'hellox', b'hello'))
        assert not b'hello'.endswith(())
        assert b'helloworld'.endswith((b'hellowo', b'rld', b'lowo'), 3)
        assert not b'helloworld'.endswith((b'hellowo', b'ello', b'rld'), 3, -1)
        assert b'hello'.endswith((b'hell', b'ell'), 0, -1)
        assert not b'hello'.endswith((b'he', b'hel'), 0, 1)
        assert b'hello'.endswith((b'he', b'hell'), 0, 4)
        raises(TypeError, b'hello'.endswith, (42,))

    def test_expandtabs(self):
        import sys

        assert b'abc\rab\tdef\ng\thi'.expandtabs() ==    b'abc\rab      def\ng       hi'
        assert b'abc\rab\tdef\ng\thi'.expandtabs(8) ==   b'abc\rab      def\ng       hi'
        assert b'abc\rab\tdef\ng\thi'.expandtabs(4) ==   b'abc\rab  def\ng   hi'
        assert b'abc\r\nab\tdef\ng\thi'.expandtabs(4) == b'abc\r\nab  def\ng   hi'
        assert b'abc\rab\tdef\ng\thi'.expandtabs() ==    b'abc\rab      def\ng       hi'
        assert b'abc\rab\tdef\ng\thi'.expandtabs(8) ==   b'abc\rab      def\ng       hi'
        assert b'abc\r\nab\r\ndef\ng\r\nhi'.expandtabs(4) == b'abc\r\nab\r\ndef\ng\r\nhi'

        s = b'xy\t'
        assert s.expandtabs() == b'xy      '

        s = b'\txy\t'
        assert s.expandtabs() == b'        xy      '
        assert s.expandtabs(1) == b' xy '
        assert s.expandtabs(2) == b'  xy  '
        assert s.expandtabs(3) == b'   xy '

        assert b'xy'.expandtabs() == b'xy'
        assert b''.expandtabs() == b''

        raises(OverflowError, b"t\tt\t".expandtabs, sys.maxint)

    def test_expandtabs_overflows_gracefully(self):
        import sys
        if sys.maxint > (1 << 32):
            skip("Wrong platform")
        raises((MemoryError, OverflowError), b't\tt\t'.expandtabs, sys.maxint)

    def test_splitlines(self):
        s = b""
        assert s.splitlines() == []
        assert s.splitlines() == s.splitlines(1)
        s = b"a + 4"
        assert s.splitlines() == [b'a + 4']
        # The following is true if no newline in string.
        assert s.splitlines() == s.splitlines(1)
        s = b"a + 4\nb + 2"
        assert s.splitlines() == [b'a + 4', b'b + 2']
        assert s.splitlines(1) == [b'a + 4\n', b'b + 2']
        s = b"ab\nab\n \n  x\n\n\n"
        assert s.splitlines() ==[b'ab',    b'ab',  b' ',   b'  x',   b'',    b'']
        assert s.splitlines() ==s.splitlines(0)
        assert s.splitlines(1) ==[b'ab\n', b'ab\n', b' \n', b'  x\n', b'\n', b'\n']
        s = b"\none\n\two\nthree\n\n"
        assert s.splitlines() ==[b'', b'one', b'\two', b'three', b'']
        assert s.splitlines(1) ==[b'\n', b'one\n', b'\two\n', b'three\n', b'\n']
        # Split on \r and \r\n too
        assert b'12\r34\r\n56'.splitlines() == [b'12', b'34', b'56']
        assert b'12\r34\r\n56'.splitlines(1) == [b'12\r', b'34\r\n', b'56']

    def test_find(self):
        assert b'abcdefghiabc'.find(b'abc') == 0
        assert b'abcdefghiabc'.find(b'abc', 1) == 9
        assert b'abcdefghiabc'.find(b'def', 4) == -1
        assert b'abcdef'.find(b'', 13) == -1
        assert b'abcdefg'.find(b'def', 5, None) == -1

    def test_index(self):
        from sys import maxint
        assert b'abcdefghiabc'.index(b'') == 0
        assert b'abcdefghiabc'.index(b'def') == 3
        assert b'abcdefghiabc'.index(b'abc') == 0
        assert b'abcdefghiabc'.index(b'abc', 1) == 9
        assert b'abcdefghiabc'.index(b'def', -4*maxint, 4*maxint) == 3
        assert b'abcdefgh'.index(b'def', 2, None) == 3
        assert b'abcdefgh'.index(b'def', None, None) == 3
        raises(ValueError, b'abcdefghiabc'.index, b'hib')
        raises(ValueError, b'abcdefghiab'.index, b'abc', 1)
        raises(ValueError, b'abcdefghi'.index, b'ghi', 8)
        raises(ValueError, b'abcdefghi'.index, b'ghi', -1)
        raises(TypeError, b'abcdefghijklmn'.index, b'abc', 0, 0.0)
        raises(TypeError, b'abcdefghijklmn'.index, b'abc', -10.0, 30)

    def test_rfind(self):
        assert b'abc'.rfind(b'', 4) == -1
        assert b'abcdefghiabc'.rfind(b'abc') == 9
        assert b'abcdefghiabc'.rfind(b'') == 12
        assert b'abcdefghiabc'.rfind(b'abcd') == 0
        assert b'abcdefghiabc'.rfind(b'abcz') == -1
        assert b'abc'.rfind(b'', 0) == 3
        assert b'abc'.rfind(b'', 3) == 3
        assert b'abcdefgh'.rfind(b'def', 2, None) == 3

    def test_rindex(self):
        from sys import maxint
        assert b'abcdefghiabc'.rindex(b'') == 12
        assert b'abcdefghiabc'.rindex(b'def') == 3
        assert b'abcdefghiabc'.rindex(b'abc') == 9
        assert b'abcdefghiabc'.rindex(b'abc', 0, -1) == 0
        assert b'abcdefghiabc'.rindex(b'abc', -4*maxint, 4*maxint) == 9
        raises(ValueError, b'abcdefghiabc'.rindex, b'hib')
        raises(ValueError, b'defghiabc'.rindex, b'def', 1)
        raises(ValueError, b'defghiabc'.rindex, b'abc', 0, -1)
        raises(ValueError, b'abcdefghi'.rindex, b'ghi', 0, 8)
        raises(ValueError, b'abcdefghi'.rindex, b'ghi', 0, -1)
        raises(TypeError, b'abcdefghijklmn'.rindex, b'abc', 0, 0.0)
        raises(TypeError, b'abcdefghijklmn'.rindex, b'abc', -10.0, 30)


    def test_partition(self):

        assert (b'this is the par', b'ti', b'tion method') == \
            b'this is the partition method'.partition(b'ti')

        # from raymond's original specification
        S = b'http://www.python.org'
        assert (b'http', b'://', b'www.python.org') == S.partition(b'://')
        assert (b'http://www.python.org', b'', b'') == S.partition(b'?')
        assert (b'', b'http://', b'www.python.org') == S.partition(b'http://')
        assert (b'http://www.python.', b'org', b'') == S.partition(b'org')

        raises(ValueError, S.partition, b'')
        raises(TypeError, S.partition, None)

    def test_rpartition(self):

        assert (b'this is the rparti', b'ti', b'on method') == \
            b'this is the rpartition method'.rpartition(b'ti')

        # from raymond's original specification
        S = b'http://www.python.org'
        assert (b'http', b'://', b'www.python.org') == S.rpartition(b'://')
        assert (b'', b'', b'http://www.python.org') == S.rpartition(b'?')
        assert (b'', b'http://', b'www.python.org') == S.rpartition(b'http://')
        assert (b'http://www.python.', b'org', b'') == S.rpartition(b'org')

        raises(ValueError, S.rpartition, b'')
        raises(TypeError, S.rpartition, None)

    def test_split_maxsplit(self):
        assert b"/a/b/c".split(b'/', 2) == [b'',b'a',b'b/c']
        assert b"a/b/c".split(b"/") == [b'a', b'b', b'c']
        assert b" a ".split(None, 0) == [b'a ']
        assert b" a ".split(None, 1) == [b'a']
        assert b" a a ".split(b" ", 0) == [b' a a ']
        assert b" a a ".split(b" ", 1) == [b'', b'a a ']

    def test_join(self):
        assert b", ".join([b'a', b'b', b'c']) == b"a, b, c"
        assert b"".join([]) == b""
        assert b"-".join([b'a', b'b']) == b'a-b'
        text = b'text'
        assert b"".join([text]) is text
        raises(TypeError, b''.join, 1)
        raises(TypeError, b''.join, [1])
        raises(TypeError, b''.join, [[1]])

    def test_unicode_join_endcase(self):
        # This class inserts a Unicode object into its argument's natural
        # iteration, in the 3rd position.
        class OhPhooey(object):
            def __init__(self, seq):
                self.it = iter(seq)
                self.i = 0

            def __iter__(self):
                return self

            def __next__(self):
                i = self.i
                self.i = i+1
                if i == 2:
                    return "fooled you!"
                return next(self.it)

        f = (b'a\n', b'b\n', b'c\n')
        raises(TypeError, b" - ".join, OhPhooey(f))

    def test_lower(self):
        assert b"aaa AAA".lower() == b"aaa aaa"
        assert b"".lower() == b""

    def test_upper(self):
        assert b"aaa AAA".upper() == b"AAA AAA"
        assert b"".upper() == b""

    def test_isalnum(self):
        assert b"".isalnum() == False
        assert b"!Bro12345w&&&&n Fox".isalnum() == False
        assert b"125 Brown Foxes".isalnum() == False
        assert b"125BrownFoxes".isalnum() == True

    def test_isalpha(self):
        assert b"".isalpha() == False
        assert b"!Bro12345w&&&&nFox".isalpha() == False
        assert b"Brown Foxes".isalpha() == False
        assert b"125".isalpha() == False

    def test_isdigit(self):
        assert b"".isdigit() == False
        assert b"!Bro12345w&&&&nFox".isdigit() == False
        assert b"Brown Foxes".isdigit() == False
        assert b"125".isdigit() == True

    def test_isspace(self):
        assert b"".isspace() == False
        assert b"!Bro12345w&&&&nFox".isspace() == False
        assert b" ".isspace() ==  True
        assert b"\t\t\b\b\n".isspace() == False
        assert b"\t\t".isspace() == True
        assert b"\t\t\r\r\n".isspace() == True

    def test_islower(self):
        assert b"".islower() == False
        assert b" ".islower() ==  False
        assert b"\t\t\b\b\n".islower() == False
        assert b"b".islower() == True
        assert b"bbb".islower() == True
        assert b"!bbb".islower() == True
        assert b"BBB".islower() == False
        assert b"bbbBBB".islower() == False

    def test_isupper(self):
        assert b"".isupper() == False
        assert b" ".isupper() ==  False
        assert b"\t\t\b\b\n".isupper() == False
        assert b"B".isupper() == True
        assert b"BBB".isupper() == True
        assert b"!BBB".isupper() == True
        assert b"bbb".isupper() == False
        assert b"BBBbbb".isupper() == False


    def test_swapcase(self):
        assert b"aaa AAA 111".swapcase() == b"AAA aaa 111"
        assert b"".swapcase() == b""

    def test_translate(self):
        def maketrans(origin, image):
            if len(origin) != len(image):
                raise ValueError("maketrans arguments must have same length")
            L = [i for i in range(256)]
            for i in range(len(origin)):
                L[origin[i]] = image[i]

            tbl = bytes(L)
            return tbl

        table = maketrans(b'abc', b'xyz')
        assert b'xyzxyz' == b'xyzabcdef'.translate(table, 'def')
        assert b'xyzxyz' == b'xyzabcdef'.translate(memoryview(table), 'def')

        table = maketrans(b'a', b'A')
        assert b'Abc' == b'abc'.translate(table)
        assert b'xyz' == b'xyz'.translate(table)
        assert b'yz' ==  b'xyz'.translate(table, 'x')

        raises(ValueError, b'xyz'.translate, b'too short', 'strip')
        raises(ValueError, b'xyz'.translate, b'too short')
        raises(ValueError, b'xyz'.translate, b'too long'*33)

        assert b'yz' == b'xyz'.translate(None, 'x')     # 2.6

    def test_iter(self):
        l=[]
        for i in iter(b"42"):
            l.append(i)
        assert l == [52, 50]

    def test_repr(self):
        assert repr(b"")       =="''"
        assert repr(b"a")      =="'a'"
        assert repr(b"'")      =='"\'"'
        assert repr(b"\'")     =="\"\'\""
        assert repr(b"\"")     =='\'"\''
        assert repr(b"\t")     =="'\\t'"
        assert repr(b"\\")     =="'\\\\'"
        assert repr(b'')       =="''"
        assert repr(b'a')      =="'a'"
        assert repr(b'"')      =="'\"'"
        assert repr(b'\'')     =='"\'"'
        assert repr(b'\"')     =="'\"'"
        assert repr(b'\t')     =="'\\t'"
        assert repr(b'\\')     =="'\\\\'"
        assert repr(b"'''\"")  =='\'\\\'\\\'\\\'"\''
        assert repr(b"\x13")   =="'\\x13'"
        assert repr(b"\x02")   =="'\\x02'"

    def test_contains(self):
        assert b'' in b'abc'
        assert b'a' in b'abc'
        assert b'ab' in b'abc'
        assert not b'd' in b'abc'
        raises(TypeError, b'a'.__contains__, 1)

    def test_decode(self):
        assert b'hello'.decode('ascii') == 'hello'

    def test_encode(self):
        assert 'hello'.encode() == b'hello'
        assert type('hello'.encode()) is bytes

    def test_hash(self):
        # check that we have the same hash as CPython for at least 31 bits
        # (but don't go checking CPython's special case -1)
        # disabled: assert hash('') == 0 --- different special case
        assert hash('hello') & 0x7fffffff == 0x347697fd
        assert hash('hello world!') & 0x7fffffff == 0x2f0bb411

    def test_buffer(self):
        x = b"he"
        x += b"llo"
        b = buffer(x)
        assert len(b) == 5
        assert b[-1] == b"o"
        assert b[:] == b"hello"
        assert b[1:0] == b""
        raises(TypeError, "b[3] = 'x'")

    def test_getnewargs(self):
        assert  b"foo".__getnewargs__() == (b"foo",)

    def test_subclass(self):
        class S(bytes):
            pass
        s = S(b'abc')
        assert type(b''.join([s])) is bytes
        assert type(s.join([])) is bytes
        assert type(s.split(b'x')[0]) is bytes
        assert type(s.ljust(3)) is bytes
        assert type(s.rjust(3)) is bytes
        assert type(S(b'A').upper()) is bytes
        assert type(S(b'a').lower()) is bytes
        assert type(S(b'A').capitalize()) is bytes
        assert type(S(b'A').title()) is bytes
        assert type(s.replace(s, s)) is bytes
        assert type(s.replace(b'x', b'y')) is bytes
        assert type(s.replace(b'x', b'y', 0)) is bytes
        assert type(s.zfill(3)) is bytes
        assert type(s.strip()) is bytes
        assert type(s.rstrip()) is bytes
        assert type(s.lstrip()) is bytes
        assert type(s.center(3)) is bytes
        assert type(s.splitlines()[0]) is bytes

    def test_replace_overflow(self):
        import sys
        if sys.maxint > 2**31-1:
            skip("Wrong platform")
        s = b"a" * (2**16)
        raises(OverflowError, s.replace, "", s)

    def test_getslice(self):
        s = b"abc"
        assert s[:] == b"abc"
        assert s[1:] == b"bc"
        assert s[:2] == b"ab"
        assert s[1:2] == b"b"
        assert s[-2:] == b"bc"
        assert s[:-1] == b"ab"
        assert s[-2:2] == b"b"
        assert s[1:-1] == b"b"
        assert s[-2:-1] == b"b"

    def test_no_len_on_str_iter(self):
        iterable = b"hello"
        raises(TypeError, len, iter(iterable))

    def test_overflow_replace(self):
        import sys
        if sys.maxint > 2**31-1:
            skip("Wrong platform")
        x = b"A" * (2**16)
        raises(OverflowError, x.replace, b'', x)

class AppTestPrebuilt(AppTestStringObject):
    def setup_class(cls):
        cls.space = gettestobjspace(**{"objspace.std.withprebuiltchar": True})

class AppTestShare(AppTestStringObject):
    def setup_class(cls):
        cls.space = gettestobjspace(**{"objspace.std.sharesmallstr": True})

class AppTestPrebuiltShare(AppTestStringObject):
    def setup_class(cls):
        cls.space = gettestobjspace(**{"objspace.std.withprebuiltchar": True,
                                       "objspace.std.sharesmallstr": True})
