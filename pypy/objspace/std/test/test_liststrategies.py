from pypy.objspace.std.listobject import W_ListObject, EmptyListStrategy, ObjectListStrategy, IntegerListStrategy, FloatListStrategy, StringListStrategy, RangeListStrategy, make_range_list
from pypy.objspace.std import listobject
from pypy.objspace.std.test.test_listobject import TestW_ListObject

from pypy.conftest import gettestobjspace

class TestW_ListStrategies(TestW_ListObject):

    def test_check_strategy(self):
        assert isinstance(W_ListObject(self.space, []).strategy, EmptyListStrategy)
        assert isinstance(W_ListObject(self.space, [self.space.wrap(1),self.space.wrap('a')]).strategy, ObjectListStrategy)
        assert isinstance(W_ListObject(self.space, [self.space.wrap(1),self.space.wrap(2),self.space.wrap(3)]).strategy, IntegerListStrategy)
        assert isinstance(W_ListObject(self.space, [self.space.wrap('a'), self.space.wrap('b')]).strategy, StringListStrategy)

    def test_empty_to_any(self):
        l = W_ListObject(self.space, [])
        assert isinstance(l.strategy, EmptyListStrategy)
        l.append(self.space.wrap((1,3)))
        assert isinstance(l.strategy, ObjectListStrategy)

        l = W_ListObject(self.space, [])
        assert isinstance(l.strategy, EmptyListStrategy)
        l.append(self.space.wrap(1))
        assert isinstance(l.strategy, IntegerListStrategy)

        l = W_ListObject(self.space, [])
        assert isinstance(l.strategy, EmptyListStrategy)
        l.append(self.space.wrap('a'))
        assert isinstance(l.strategy, StringListStrategy)

        l = W_ListObject(self.space, [])
        assert isinstance(l.strategy, EmptyListStrategy)
        l.append(self.space.wrap(1.2))
        assert isinstance(l.strategy, FloatListStrategy)

    def test_int_to_any(self):
        l = W_ListObject(self.space, [self.space.wrap(1),self.space.wrap(2),self.space.wrap(3)])
        assert isinstance(l.strategy, IntegerListStrategy)
        l.append(self.space.wrap(4))
        assert isinstance(l.strategy, IntegerListStrategy)
        l.append(self.space.wrap('a'))
        assert isinstance(l.strategy, ObjectListStrategy)

    def test_string_to_any(self):
        l = W_ListObject(self.space, [self.space.wrap('a'),self.space.wrap('b'),self.space.wrap('c')])
        assert isinstance(l.strategy, StringListStrategy)
        l.append(self.space.wrap('d'))
        assert isinstance(l.strategy, StringListStrategy)
        l.append(self.space.wrap(3))
        assert isinstance(l.strategy, ObjectListStrategy)

    def test_float_to_any(self):
        l = W_ListObject(self.space, [self.space.wrap(1.1),self.space.wrap(2.2),self.space.wrap(3.3)])
        assert isinstance(l.strategy, FloatListStrategy)
        l.append(self.space.wrap(4.4))
        assert isinstance(l.strategy, FloatListStrategy)
        l.append(self.space.wrap("a"))
        assert isinstance(l.strategy, ObjectListStrategy)

    def test_setitem(self):
        # This should work if test_listobject.py passes
        l = W_ListObject(self.space, [self.space.wrap('a'),self.space.wrap('b'),self.space.wrap('c')])
        assert self.space.eq_w(l.getitem(0), self.space.wrap('a'))
        l.setitem(0, self.space.wrap('d'))
        assert self.space.eq_w(l.getitem(0), self.space.wrap('d'))

        assert isinstance(l.strategy, StringListStrategy)

        # IntStrategy to ObjectStrategy
        l = W_ListObject(self.space, [self.space.wrap(1),self.space.wrap(2),self.space.wrap(3)])
        assert isinstance(l.strategy, IntegerListStrategy)
        l.setitem(0, self.space.wrap('d'))
        assert isinstance(l.strategy, ObjectListStrategy)

        # StringStrategy to ObjectStrategy
        l = W_ListObject(self.space, [self.space.wrap('a'),self.space.wrap('b'),self.space.wrap('c')])
        assert isinstance(l.strategy, StringListStrategy)
        l.setitem(0, self.space.wrap(2))
        assert isinstance(l.strategy, ObjectListStrategy)

        # FloatStrategy to ObjectStrategy
        l = W_ListObject(self.space, [self.space.wrap(1.2),self.space.wrap(2.3),self.space.wrap(3.4)])
        assert isinstance(l.strategy, FloatListStrategy)
        l.setitem(0, self.space.wrap("a"))
        assert isinstance(l.strategy, ObjectListStrategy)

    def test_insert(self):
        # no change
        l = W_ListObject(self.space, [self.space.wrap(1),self.space.wrap(2),self.space.wrap(3)])
        assert isinstance(l.strategy, IntegerListStrategy)
        l.insert(3, self.space.wrap(4))
        assert isinstance(l.strategy, IntegerListStrategy)

        # StringStrategy
        l = W_ListObject(self.space, [self.space.wrap('a'),self.space.wrap('b'),self.space.wrap('c')])
        assert isinstance(l.strategy, StringListStrategy)
        l.insert(3, self.space.wrap(2))
        assert isinstance(l.strategy, ObjectListStrategy)

        # IntegerStrategy
        l = W_ListObject(self.space, [self.space.wrap(1),self.space.wrap(2),self.space.wrap(3)])
        assert isinstance(l.strategy, IntegerListStrategy)
        l.insert(3, self.space.wrap('d'))
        assert isinstance(l.strategy, ObjectListStrategy)

        # FloatStrategy
        l = W_ListObject(self.space, [self.space.wrap(1.1),self.space.wrap(2.2),self.space.wrap(3.3)])
        assert isinstance(l.strategy, FloatListStrategy)
        l.insert(3, self.space.wrap('d'))
        assert isinstance(l.strategy, ObjectListStrategy)

        # EmptyStrategy
        l = W_ListObject(self.space, [])
        assert isinstance(l.strategy, EmptyListStrategy)
        l.insert(0, self.space.wrap('a'))
        assert isinstance(l.strategy, StringListStrategy)

        l = W_ListObject(self.space, [])
        assert isinstance(l.strategy, EmptyListStrategy)
        l.insert(0, self.space.wrap(2))
        assert isinstance(l.strategy, IntegerListStrategy)

    def test_list_empty_after_delete(self):
        import py
        py.test.skip("return to emptyliststrategy is not supported anymore")
        l = W_ListObject(self.space, [self.space.wrap(3)])
        assert isinstance(l.strategy, IntegerListStrategy)
        l.deleteitem(0)
        assert isinstance(l.strategy, EmptyListStrategy)

        l = W_ListObject(self.space, [self.space.wrap(1), self.space.wrap(2)])
        assert isinstance(l.strategy, IntegerListStrategy)
        l.deleteslice(0, 1, 2)
        assert isinstance(l.strategy, EmptyListStrategy)

        l = W_ListObject(self.space, [self.space.wrap(1)])
        assert isinstance(l.strategy, IntegerListStrategy)
        l.pop(-1)
        assert isinstance(l.strategy, EmptyListStrategy)

    def test_setslice(self):
        l = W_ListObject(self.space, [])
        assert isinstance(l.strategy, EmptyListStrategy)
        l.setslice(0, 1, 2, W_ListObject(self.space, [self.space.wrap(1), self.space.wrap(2), self.space.wrap(3)]))
        assert isinstance(l.strategy, IntegerListStrategy)

        # IntegerStrategy to IntegerStrategy
        l = W_ListObject(self.space, [self.space.wrap(1), self.space.wrap(2), self.space.wrap(3)])
        assert isinstance(l.strategy, IntegerListStrategy)
        l.setslice(0, 1, 2, W_ListObject(self.space, [self.space.wrap(4), self.space.wrap(5), self.space.wrap(6)]))
        assert isinstance(l.strategy, IntegerListStrategy)

        # ObjectStrategy to ObjectStrategy
        l = W_ListObject(self.space, [self.space.wrap(1), self.space.wrap('b'), self.space.wrap(3)])
        assert isinstance(l.strategy, ObjectListStrategy)
        l.setslice(0, 1, 2, W_ListObject(self.space, [self.space.wrap(1), self.space.wrap(2), self.space.wrap(3)]))
        assert isinstance(l.strategy, ObjectListStrategy)

        # IntegerStrategy to ObjectStrategy
        l = W_ListObject(self.space, [self.space.wrap(1), self.space.wrap(2), self.space.wrap(3)])
        assert isinstance(l.strategy, IntegerListStrategy)
        l.setslice(0, 1, 2, W_ListObject(self.space, [self.space.wrap('a'), self.space.wrap('b'), self.space.wrap('c')]))
        assert isinstance(l.strategy, ObjectListStrategy)

        # StringStrategy to ObjectStrategy
        l = W_ListObject(self.space, [self.space.wrap('a'), self.space.wrap('b'), self.space.wrap('c')])
        assert isinstance(l.strategy, StringListStrategy)
        l.setslice(0, 1, 2, W_ListObject(self.space, [self.space.wrap(1), self.space.wrap(2), self.space.wrap(3)]))
        assert isinstance(l.strategy, ObjectListStrategy)

        # FloatStrategy to ObjectStrategy
        l = W_ListObject(self.space, [self.space.wrap(1.1), self.space.wrap(2.2), self.space.wrap(3.3)])
        assert isinstance(l.strategy, FloatListStrategy)
        l.setslice(0, 1, 2, W_ListObject(self.space, [self.space.wrap('a'), self.space.wrap(2), self.space.wrap(3)]))
        assert isinstance(l.strategy, ObjectListStrategy)

    def test_setslice_List(self):

        def wrapitems(items):
            items_w = []
            for i in items:
                items_w.append(self.space.wrap(i))
            return items_w

        def keep_other_strategy(w_list, start, step, length, w_other):
            other_strategy = w_other.strategy
            w_list.setslice(start, step, length, w_other)
            assert w_other.strategy is other_strategy

        l = W_ListObject(self.space, wrapitems([1,2,3,4,5]))
        other = W_ListObject(self.space, wrapitems(["a", "b", "c"]))
        keep_other_strategy(l, 0, 2, other.length(), other)
        assert l.strategy is self.space.fromcache(ObjectListStrategy)

        l = W_ListObject(self.space, wrapitems([1,2,3,4,5]))
        other = W_ListObject(self.space, wrapitems([6, 6, 6]))
        keep_other_strategy(l, 0, 2, other.length(), other)
        assert l.strategy is self.space.fromcache(IntegerListStrategy)

        l = W_ListObject(self.space, wrapitems(["a","b","c","d","e"]))
        other = W_ListObject(self.space, wrapitems(["a", "b", "c"]))
        keep_other_strategy(l, 0, 2, other.length(), other)
        assert l.strategy is self.space.fromcache(StringListStrategy)

        l = W_ListObject(self.space, wrapitems([1.1, 2.2, 3.3, 4.4, 5.5]))
        other = W_ListObject(self.space, [])
        keep_other_strategy(l, 0, 1, l.length(), other)
        assert l.strategy is self.space.fromcache(FloatListStrategy)

        l = W_ListObject(self.space, wrapitems(["a",3,"c",4,"e"]))
        other = W_ListObject(self.space, wrapitems(["a", "b", "c"]))
        keep_other_strategy(l, 0, 2, other.length(), other)
        assert l.strategy is self.space.fromcache(ObjectListStrategy)

        l = W_ListObject(self.space, wrapitems(["a",3,"c",4,"e"]))
        other = W_ListObject(self.space, [])
        keep_other_strategy(l, 0, 1, l.length(), other)
        assert l.strategy is self.space.fromcache(ObjectListStrategy)

    def test_empty_setslice_with_objectlist(self):
        l = W_ListObject(self.space, [])
        o = W_ListObject(self.space, [self.space.wrap(1), self.space.wrap("2"), self.space.wrap(3)])
        l.setslice(0, 1, o.length(), o)
        assert l.getitems() == o.getitems()
        l.append(self.space.wrap(17))
        assert l.getitems() != o.getitems()

    def test_extend(self):
        l = W_ListObject(self.space, [])
        assert isinstance(l.strategy, EmptyListStrategy)
        l.extend(W_ListObject(self.space, [self.space.wrap(1), self.space.wrap(2), self.space.wrap(3)]))
        assert isinstance(l.strategy, IntegerListStrategy)

        l = W_ListObject(self.space, [self.space.wrap(1), self.space.wrap(2), self.space.wrap(3)])
        assert isinstance(l.strategy, IntegerListStrategy)
        l.extend(W_ListObject(self.space, [self.space.wrap('a'), self.space.wrap('b'), self.space.wrap('c')]))
        assert isinstance(l.strategy, ObjectListStrategy)

        l = W_ListObject(self.space, [self.space.wrap(1), self.space.wrap(2), self.space.wrap(3)])
        assert isinstance(l.strategy, IntegerListStrategy)
        l.extend(W_ListObject(self.space, [self.space.wrap(4), self.space.wrap(5), self.space.wrap(6)]))
        assert isinstance(l.strategy, IntegerListStrategy)

        l = W_ListObject(self.space, [self.space.wrap(1.1), self.space.wrap(2.2), self.space.wrap(3.3)])
        assert isinstance(l.strategy, FloatListStrategy)
        l.extend(W_ListObject(self.space, [self.space.wrap(4), self.space.wrap(5), self.space.wrap(6)]))
        assert isinstance(l.strategy, ObjectListStrategy)

    def test_empty_extend_with_any(self):
        empty = W_ListObject(self.space, [])
        assert isinstance(empty.strategy, EmptyListStrategy)
        empty.extend(W_ListObject(self.space, [self.space.wrap(1), self.space.wrap(2), self.space.wrap(3)]))
        assert isinstance(empty.strategy, IntegerListStrategy)

        empty = W_ListObject(self.space, [])
        assert isinstance(empty.strategy, EmptyListStrategy)
        empty.extend(W_ListObject(self.space, [self.space.wrap("a"), self.space.wrap("b"), self.space.wrap("c")]))
        assert isinstance(empty.strategy, StringListStrategy)

        empty = W_ListObject(self.space, [])
        assert isinstance(empty.strategy, EmptyListStrategy)
        r = make_range_list(self.space, 1,3,7)
        empty.extend(r)
        assert isinstance(empty.strategy, RangeListStrategy)
        print empty.getitem(6)
        assert self.space.is_true(self.space.eq(empty.getitem(1), self.space.wrap(4)))

        empty = W_ListObject(self.space, [])
        assert isinstance(empty.strategy, EmptyListStrategy)
        empty.extend(W_ListObject(self.space, [self.space.wrap(1), self.space.wrap(2), self.space.wrap(3)]))
        assert isinstance(empty.strategy, IntegerListStrategy)

        empty = W_ListObject(self.space, [])
        assert isinstance(empty.strategy, EmptyListStrategy)
        empty.extend(W_ListObject(self.space, [self.space.wrap(1.1), self.space.wrap(2.2), self.space.wrap(3.3)]))
        assert isinstance(empty.strategy, FloatListStrategy)

        empty = W_ListObject(self.space, [])
        assert isinstance(empty.strategy, EmptyListStrategy)
        empty.extend(W_ListObject(self.space, []))
        assert isinstance(empty.strategy, EmptyListStrategy)

    def test_extend_other_with_empty(self):
        l = W_ListObject(self.space, [self.space.wrap(1), self.space.wrap(2), self.space.wrap(3)])
        assert isinstance(l.strategy, IntegerListStrategy)
        l.extend(W_ListObject(self.space, []))
        assert isinstance(l.strategy, IntegerListStrategy)

    def test_rangelist(self):
        l = make_range_list(self.space, 1,3,7)
        assert isinstance(l.strategy, RangeListStrategy)
        v = l.pop(5)
        assert self.space.eq_w(v, self.space.wrap(16))
        assert isinstance(l.strategy, IntegerListStrategy)

        l = make_range_list(self.space, 1,3,7)
        assert isinstance(l.strategy, RangeListStrategy)
        v = l.pop(0)
        assert self.space.eq_w(v, self.space.wrap(1))
        assert isinstance(l.strategy, RangeListStrategy)
        v = l.pop(l.length() - 1)
        assert self.space.eq_w(v, self.space.wrap(19))
        assert isinstance(l.strategy, RangeListStrategy)
        v = l.pop_end()
        assert self.space.eq_w(v, self.space.wrap(16))
        assert isinstance(l.strategy, RangeListStrategy)

        l = make_range_list(self.space, 1,3,7)
        assert isinstance(l.strategy, RangeListStrategy)
        l.append(self.space.wrap("string"))
        assert isinstance(l.strategy, ObjectListStrategy)

        l = make_range_list(self.space, 1,1,5)
        assert isinstance(l.strategy, RangeListStrategy)
        l.append(self.space.wrap(19))
        assert isinstance(l.strategy, IntegerListStrategy)

    def test_keep_range(self):
        # simple list
        l = make_range_list(self.space, 1,1,5)
        assert isinstance(l.strategy, RangeListStrategy)
        x = l.pop(0)
        assert self.space.eq_w(x, self.space.wrap(1))
        assert isinstance(l.strategy, RangeListStrategy)
        l.pop(l.length()-1)
        assert isinstance(l.strategy, RangeListStrategy)
        l.append(self.space.wrap(5))
        assert isinstance(l.strategy, RangeListStrategy)

        # complex list
        l = make_range_list(self.space, 1,3,5)
        assert isinstance(l.strategy, RangeListStrategy)
        l.append(self.space.wrap(16))
        assert isinstance(l.strategy, RangeListStrategy)

    def test_empty_range(self):
        l = make_range_list(self.space, 0, 0, 0)
        assert isinstance(l.strategy, EmptyListStrategy)

        l = make_range_list(self.space, 1, 1, 10)
        for i in l.getitems():
            assert isinstance(l.strategy, RangeListStrategy)
            l.pop(l.length()-1)

        assert isinstance(l.strategy, RangeListStrategy)

    def test_range_setslice(self):
        l = make_range_list(self.space, 1, 3, 5)
        assert isinstance(l.strategy, RangeListStrategy)
        l.setslice(0, 1, 3, W_ListObject(self.space, [self.space.wrap(1), self.space.wrap(2), self.space.wrap(3)]))
        assert isinstance(l.strategy, IntegerListStrategy)

    def test_copy_list(self):
        l1 = W_ListObject(self.space, [self.space.wrap(1), self.space.wrap(2), self.space.wrap(3)])
        l2 = l1.clone()
        l2.append(self.space.wrap(4))
        assert not l2 == l1.getitems()

    def test_getitems_does_not_copy_object_list(self):
        l1 = W_ListObject(self.space, [self.space.wrap(1), self.space.wrap("two"), self.space.wrap(3)])
        l2 = l1.getitems()
        l2.append(self.space.wrap("four"))
        assert l2 == l1.getitems()

    def test_clone(self):
        l1 = W_ListObject(self.space, [self.space.wrap(1), self.space.wrap(2), self.space.wrap(3)])
        clone = l1.clone()
        assert isinstance(clone.strategy, IntegerListStrategy)
        clone.append(self.space.wrap(7))
        assert not self.space.eq_w(l1, clone)

    def test_add_does_not_use_getitems(self):
        l1 = W_ListObject(self.space, [self.space.wrap(1), self.space.wrap(2), self.space.wrap(3)])
        l1.getitems = None
        l2 = W_ListObject(self.space, [self.space.wrap(1), self.space.wrap(2), self.space.wrap(3)])
        l2.getitems = None
        l3 = self.space.add(l1, l2)
        l4 = W_ListObject(self.space, [self.space.wrap(1), self.space.wrap(2), self.space.wrap(3), self.space.wrap(1), self.space.wrap(2), self.space.wrap(3)])
        assert self.space.eq_w(l3, l4)

    def test_mul(self):
        l1 = W_ListObject(self.space, [self.space.wrap(1), self.space.wrap(2), self.space.wrap(3)])
        l2 = l1.mul(2)
        l3 = W_ListObject(self.space, [self.space.wrap(1), self.space.wrap(2), self.space.wrap(3), self.space.wrap(1), self.space.wrap(2), self.space.wrap(3)])
        assert self.space.eq_w(l2, l3)

        l4 = make_range_list(self.space, 1, 1, 3)
        assert self.space.eq_w(l4, l1)

        l5 = l4.mul(2)
        assert self.space.eq_w(l5, l3)

    def test_mul_same_strategy_but_different_object(self):
        l1 = W_ListObject(self.space, [self.space.wrap(1), self.space.wrap(2), self.space.wrap(3)])
        l2 = l1.mul(1)
        assert self.space.eq_w(l1, l2)
        l1.setitem(0, self.space.wrap(5))
        assert not self.space.eq_w(l1, l2)

    def test_weird_rangelist_bug(self):
        l = make_range_list(self.space, 1, 1, 3)
        from pypy.objspace.std.listobject import getslice__List_ANY_ANY
        # should not raise
        assert getslice__List_ANY_ANY(self.space, l, self.space.wrap(15), self.space.wrap(2222)).strategy == self.space.fromcache(EmptyListStrategy)

    def test_add_to_rangelist(self):
        l1 = make_range_list(self.space, 1, 1, 3)
        l2 = W_ListObject(self.space, [self.space.wrap(4), self.space.wrap(5)])
        from pypy.objspace.std.listobject import add__List_List
        l3 = add__List_List(self.space, l1, l2)
        assert self.space.eq_w(l3, W_ListObject(self.space, [self.space.wrap(1), self.space.wrap(2), self.space.wrap(3), self.space.wrap(4), self.space.wrap(5)]))

    def test_unicode(self):
        l1 = W_ListObject(self.space, [self.space.wrap("eins"), self.space.wrap("zwei")])
        assert isinstance(l1.strategy, StringListStrategy)
        l2 = W_ListObject(self.space, [self.space.wrap(u"eins"), self.space.wrap(u"zwei")])
        assert isinstance(l2.strategy, ObjectListStrategy)
        l3 = W_ListObject(self.space, [self.space.wrap("eins"), self.space.wrap(u"zwei")])
        assert isinstance(l3.strategy, ObjectListStrategy)

    def test_listview_str(self):
        space = self.space
        assert space.listview_str(space.wrap("a")) is None
        w_l = self.space.newlist([self.space.wrap('a'), self.space.wrap('b')])
        assert space.listview_str(w_l) == ["a", "b"]

    def test_string_join_uses_listview_str(self):
        space = self.space
        w_l = self.space.newlist([self.space.wrap('a'), self.space.wrap('b')])
        w_l.getitems = None
        assert space.str_w(space.call_method(space.wrap("c"), "join", w_l)) == "acb"

    def test_string_join_returns_same_instance(self):
        space = self.space
        w_text = space.wrap("text")
        w_l = self.space.newlist([w_text])
        w_l.getitems = None
        assert space.is_w(space.call_method(space.wrap(" -- "), "join", w_l), w_text)

    def test_newlist_str(self):
        space = self.space
        l = ['a', 'b']
        w_l = self.space.newlist_str(l)
        assert isinstance(w_l.strategy, StringListStrategy)
        assert space.listview_str(w_l) is l

    def test_string_uses_newlist_str(self):
        space = self.space
        w_s = space.wrap("a b c")
        space.newlist = None
        try:
            w_l = space.call_method(w_s, "split")
            w_l2 = space.call_method(w_s, "split", space.wrap(" "))
        finally:
            del space.newlist
        assert space.listview_str(w_l) == ["a", "b", "c"]
        assert space.listview_str(w_l2) == ["a", "b", "c"]

    def test_pop_without_argument_is_fast(self):
        space = self.space
        w_l = W_ListObject(space, [space.wrap(1), space.wrap(2), space.wrap(3)])
        w_l.pop = None
        w_res = listobject.list_pop__List_ANY(space, w_l, space.w_None) # does not crash
        assert space.unwrap(w_res) == 3


class TestW_ListStrategiesDisabled:
    def setup_class(cls):
        cls.space = gettestobjspace(**{"objspace.std.withliststrategies" :
                                       False})

    def test_check_strategy(self):
        assert isinstance(W_ListObject(self.space, []).strategy, ObjectListStrategy)
        assert isinstance(W_ListObject(self.space, [self.space.wrap(1),self.space.wrap('a')]).strategy, ObjectListStrategy)
        assert isinstance(W_ListObject(self.space, [self.space.wrap(1),self.space.wrap(2),self.space.wrap(3)]).strategy, ObjectListStrategy)
        assert isinstance(W_ListObject(self.space, [self.space.wrap('a'), self.space.wrap('b')]).strategy, ObjectListStrategy)
