from pypy.objspace.std.setobject import W_SetObject
from pypy.objspace.std.setobject import IntegerSetStrategy, ObjectSetStrategy, EmptySetStrategy
from pypy.objspace.std.listobject import W_ListObject

class TestW_SetStrategies:

    def wrapped(self, l):
        return W_ListObject(self.space, [self.space.wrap(x) for x in l])

    def test_from_list(self):
        s = W_SetObject(self.space, self.wrapped([1,2,3,4,5]))
        assert s.strategy is self.space.fromcache(IntegerSetStrategy)

        s = W_SetObject(self.space, self.wrapped([1,"two",3,"four",5]))
        assert s.strategy is self.space.fromcache(ObjectSetStrategy)

        s = W_SetObject(self.space)
        assert s.strategy is self.space.fromcache(EmptySetStrategy)

        s = W_SetObject(self.space, self.wrapped([]))
        assert s.strategy is self.space.fromcache(EmptySetStrategy)

    def test_switch_to_object(self):
        s = W_SetObject(self.space, self.wrapped([1,2,3,4,5]))
        s.add(self.space.wrap("six"))
        assert s.strategy is self.space.fromcache(ObjectSetStrategy)

        s1 = W_SetObject(self.space, self.wrapped([1,2,3,4,5]))
        s2 = W_SetObject(self.space, self.wrapped(["six", "seven"]))
        s1.update(s2)
        assert s1.strategy is self.space.fromcache(ObjectSetStrategy)

    def test_symmetric_difference(self):
        s1 = W_SetObject(self.space, self.wrapped([1,2,3,4,5]))
        s2 = W_SetObject(self.space, self.wrapped(["six", "seven"]))
        s1.symmetric_difference_update(s2)
        assert s1.strategy is self.space.fromcache(ObjectSetStrategy)

    def test_intersection(self):
        s1 = W_SetObject(self.space, self.wrapped([1,2,3,4,5]))
        s2 = W_SetObject(self.space, self.wrapped([4,5, "six", "seven"]))
        s3 = s1.intersect(s2)
        skip("for now intersection with ObjectStrategy always results in another ObjectStrategy")
        assert s3.strategy is self.space.fromcache(IntegerSetStrategy)

    def test_clear(self):
        s1 = W_SetObject(self.space, self.wrapped([1,2,3,4,5]))
        s1.clear()
        assert s1.strategy is self.space.fromcache(EmptySetStrategy)

    def test_remove(self):
        from pypy.objspace.std.setobject import set_remove__Set_ANY
        s1 = W_SetObject(self.space, self.wrapped([1]))
        set_remove__Set_ANY(self.space, s1, self.space.wrap(1))
        assert s1.strategy is self.space.fromcache(EmptySetStrategy)

    def test_union(self):
        from pypy.objspace.std.setobject import set_union__Set
        s1 = W_SetObject(self.space, self.wrapped([1,2,3,4,5]))
        s2 = W_SetObject(self.space, self.wrapped([4,5,6,7]))
        s3 = W_SetObject(self.space, self.wrapped([4,'5','6',7]))
        s4 = set_union__Set(self.space, s1, [s2])
        s5 = set_union__Set(self.space, s1, [s3])
        assert s4.strategy is self.space.fromcache(IntegerSetStrategy)
        assert s5.strategy is self.space.fromcache(ObjectSetStrategy)

    def test_discard(self):
        class FakeInt(object):
            def __init__(self, value):
                self.value = value
            def __hash__(self):
                return hash(self.value)
            def __eq__(self, other):
                if other == self.value:
                    return True
                return False

        from pypy.objspace.std.setobject import set_discard__Set_ANY

        s1 = W_SetObject(self.space, self.wrapped([1,2,3,4,5]))
        set_discard__Set_ANY(self.space, s1, self.space.wrap("five"))
        skip("currently not supported")
        assert s1.strategy is self.space.fromcache(IntegerSetStrategy)

        set_discard__Set_ANY(self.space, s1, self.space.wrap(FakeInt(5)))
        assert s1.strategy is self.space.fromcache(ObjectSetStrategy)

    def test_has_key(self):
        class FakeInt(object):
            def __init__(self, value):
                self.value = value
            def __hash__(self):
                return hash(self.value)
            def __eq__(self, other):
                if other == self.value:
                    return True
                return False

        from pypy.objspace.std.setobject import set_discard__Set_ANY

        s1 = W_SetObject(self.space, self.wrapped([1,2,3,4,5]))
        assert not s1.has_key(self.space.wrap("five"))
        skip("currently not supported")
        assert s1.strategy is self.space.fromcache(IntegerSetStrategy)

        assert s1.has_key(self.space.wrap(FakeInt(2)))
        assert s1.strategy is self.space.fromcache(ObjectSetStrategy)
