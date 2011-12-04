
import py
from pypy.module.micronumpy.test.test_base import BaseNumpyAppTest
from pypy.module.micronumpy.interp_numarray import W_NDimArray, shape_agreement
from pypy.module.micronumpy import signature
from pypy.interpreter.error import OperationError
from pypy.conftest import gettestobjspace


class MockDtype(object):
    signature = signature.BaseSignature()

    def malloc(self, size):
        return None


class TestNumArrayDirect(object):
    def newslice(self, *args):
        return self.space.newslice(*[self.space.wrap(arg) for arg in args])

    def newtuple(self, *args):
        args_w = []
        for arg in args:
            if isinstance(arg, int):
                args_w.append(self.space.wrap(arg))
            else:
                args_w.append(arg)
        return self.space.newtuple(args_w)

    def test_strides_f(self):
        a = W_NDimArray(100, [10, 5, 3], MockDtype(), 'F')
        assert a.strides == [1, 10, 50]
        assert a.backstrides == [9, 40, 100]

    def test_strides_c(self):
        a = W_NDimArray(100, [10, 5, 3], MockDtype(), 'C')
        assert a.strides == [15, 3, 1]
        assert a.backstrides == [135, 12, 2]

    def test_create_slice_f(self):
        space = self.space
        a = W_NDimArray(10 * 5 * 3, [10, 5, 3], MockDtype(), 'F')
        s = a.create_slice(space, [(3, 0, 0, 1)])
        assert s.start == 3
        assert s.strides == [10, 50]
        assert s.backstrides == [40, 100]
        s = a.create_slice(space, [(1, 9, 2, 4)])
        assert s.start == 1
        assert s.strides == [2, 10, 50]
        assert s.backstrides == [6, 40, 100]
        s = a.create_slice(space, [(1, 5, 3, 2), (1, 2, 1, 1), (1, 0, 0, 1)])
        assert s.shape == [2, 1]
        assert s.strides == [3, 10]
        assert s.backstrides == [3, 0]
        s = a.create_slice(space, [(0, 10, 1, 10), (2, 0, 0, 1)])
        assert s.start == 20
        assert s.shape == [10, 3]

    def test_create_slice_c(self):
        space = self.space
        a = W_NDimArray(10 * 5 * 3, [10, 5, 3], MockDtype(), 'C')
        s = a.create_slice(space, [(3, 0, 0, 1)])
        assert s.start == 45
        assert s.strides == [3, 1]
        assert s.backstrides == [12, 2]
        s = a.create_slice(space, [(1, 9, 2, 4)])
        assert s.start == 15
        assert s.strides == [30, 3, 1]
        assert s.backstrides == [90, 12, 2]
        s = a.create_slice(space, [(1, 5, 3, 2), (1, 2, 1, 1), (1, 0, 0, 1)])
        assert s.start == 19
        assert s.shape == [2, 1]
        assert s.strides == [45, 3]
        assert s.backstrides == [45, 0]
        s = a.create_slice(space, [(0, 10, 1, 10), (2, 0, 0, 1)])
        assert s.start == 6
        assert s.shape == [10, 3]

    def test_slice_of_slice_f(self):
        space = self.space
        a = W_NDimArray(10 * 5 * 3, [10, 5, 3], MockDtype(), 'F')
        s = a.create_slice(space, [(5, 0, 0, 1)])
        assert s.start == 5
        s2 = s.create_slice(space, [(3, 0, 0, 1)])
        assert s2.shape == [3]
        assert s2.strides == [50]
        assert s2.parent is a
        assert s2.backstrides == [100]
        assert s2.start == 35
        s = a.create_slice(space, [(1, 5, 3, 2)])
        s2 = s.create_slice(space, [(0, 2, 1, 2), (2, 0, 0, 1)])
        assert s2.shape == [2, 3]
        assert s2.strides == [3, 50]
        assert s2.backstrides == [3, 100]
        assert s2.start == 1 * 15 + 2 * 3

    def test_slice_of_slice_c(self):
        space = self.space
        a = W_NDimArray(10 * 5 * 3, [10, 5, 3], MockDtype(), order='C')
        s = a.create_slice(space, [(5, 0, 0, 1)])
        assert s.start == 15 * 5
        s2 = s.create_slice(space, [(3, 0, 0, 1)])
        assert s2.shape == [3]
        assert s2.strides == [1]
        assert s2.parent is a
        assert s2.backstrides == [2]
        assert s2.start == 5 * 15 + 3 * 3
        s = a.create_slice(space, [(1, 5, 3, 2)])
        s2 = s.create_slice(space, [(0, 2, 1, 2), (2, 0, 0, 1)])
        assert s2.shape == [2, 3]
        assert s2.strides == [45, 1]
        assert s2.backstrides == [45, 2]
        assert s2.start == 1 * 15 + 2 * 3

    def test_negative_step_f(self):
        space = self.space
        a = W_NDimArray(10 * 5 * 3, [10, 5, 3], MockDtype(), 'F')
        s = a.create_slice(space, [(9, -1, -2, 5)])
        assert s.start == 9
        assert s.strides == [-2, 10, 50]
        assert s.backstrides == [-8, 40, 100]

    def test_negative_step_c(self):
        space = self.space
        a = W_NDimArray(10 * 5 * 3, [10, 5, 3], MockDtype(), order='C')
        s = a.create_slice(space, [(9, -1, -2, 5)])
        assert s.start == 135
        assert s.strides == [-30, 3, 1]
        assert s.backstrides == [-120, 12, 2]

    def test_index_of_single_item_f(self):
        a = W_NDimArray(10 * 5 * 3, [10, 5, 3], MockDtype(), 'F')
        r = a._index_of_single_item(self.space, self.newtuple(1, 2, 2))
        assert r == 1 + 2 * 10 + 2 * 50
        s = a.create_slice(self.space, [(0, 10, 1, 10), (2, 0, 0, 1)])
        r = s._index_of_single_item(self.space, self.newtuple(1, 0))
        assert r == a._index_of_single_item(self.space, self.newtuple(1, 2, 0))
        r = s._index_of_single_item(self.space, self.newtuple(1, 1))
        assert r == a._index_of_single_item(self.space, self.newtuple(1, 2, 1))

    def test_index_of_single_item_c(self):
        a = W_NDimArray(10 * 5 * 3, [10, 5, 3], MockDtype(), 'C')
        r = a._index_of_single_item(self.space, self.newtuple(1, 2, 2))
        assert r == 1 * 3 * 5 + 2 * 3 + 2
        s = a.create_slice(self.space, [(0, 10, 1, 10), (2, 0, 0, 1)])
        r = s._index_of_single_item(self.space, self.newtuple(1, 0))
        assert r == a._index_of_single_item(self.space, self.newtuple(1, 2, 0))
        r = s._index_of_single_item(self.space, self.newtuple(1, 1))
        assert r == a._index_of_single_item(self.space, self.newtuple(1, 2, 1))

    def test_shape_agreement(self):
        assert shape_agreement(self.space, [3], [3]) == [3]
        assert shape_agreement(self.space, [1, 2, 3], [1, 2, 3]) == [1, 2, 3]
        py.test.raises(OperationError, shape_agreement, self.space, [2], [3])
        assert shape_agreement(self.space, [4, 4], []) == [4, 4]
        assert shape_agreement(self.space,
                [8, 1, 6, 1], [7, 1, 5]) == [8, 7, 6, 5]
        assert shape_agreement(self.space,
                [5, 2], [4, 3, 5, 2]) == [4, 3, 5, 2]


class AppTestNumArray(BaseNumpyAppTest):
    def test_ndarray(self):
        from numpypy import ndarray, array, dtype

        assert type(ndarray) is type
        assert type(array) is not type
        a = ndarray((2, 3))
        assert a.shape == (2, 3)
        assert a.dtype == dtype(float)

        raises(TypeError, ndarray, [[1], [2], [3]])

        a = ndarray(3, dtype=int)
        assert a.shape == (3,)
        assert a.dtype is dtype(int)

    def test_type(self):
        from numpypy import array
        ar = array(range(5))
        assert type(ar) is type(ar + ar)

    def test_init(self):
        from numpypy import zeros
        a = zeros(15)
        # Check that storage was actually zero'd.
        assert a[10] == 0.0
        # And check that changes stick.
        a[13] = 5.3
        assert a[13] == 5.3

    def test_size(self):
        from numpypy import array
        assert array(3).size == 1
        a = array([1, 2, 3])
        assert a.size == 3
        assert (a + a).size == 3

    def test_empty(self):
        """
        Test that empty() works.
        """

        from numpypy import empty
        a = empty(2)
        a[1] = 1.0
        assert a[1] == 1.0

    def test_ones(self):
        from numpypy import ones
        a = ones(3)
        assert len(a) == 3
        assert a[0] == 1
        raises(IndexError, "a[3]")
        a[2] = 4
        assert a[2] == 4

    def test_copy(self):
        from numpypy import array
        a = array(range(5))
        b = a.copy()
        for i in xrange(5):
            assert b[i] == a[i]
        a[3] = 22
        assert b[3] == 3

        a = array(1)
        assert a.copy() == a

    def test_iterator_init(self):
        from numpypy import array
        a = array(range(5))
        assert a[3] == 3

    def test_getitem(self):
        from numpypy import array
        a = array(range(5))
        raises(IndexError, "a[5]")
        a = a + a
        raises(IndexError, "a[5]")
        assert a[-1] == 8
        raises(IndexError, "a[-6]")

    def test_getitem_tuple(self):
        from numpypy import array
        a = array(range(5))
        raises(IndexError, "a[(1,2)]")
        for i in xrange(5):
            assert a[(i,)] == i
        b = a[()]
        for i in xrange(5):
            assert a[i] == b[i]

    def test_setitem(self):
        from numpypy import array
        a = array(range(5))
        a[-1] = 5.0
        assert a[4] == 5.0
        raises(IndexError, "a[5] = 0.0")
        raises(IndexError, "a[-6] = 3.0")

    def test_setitem_tuple(self):
        from numpypy import array
        a = array(range(5))
        raises(IndexError, "a[(1,2)] = [0,1]")
        for i in xrange(5):
            a[(i,)] = i + 1
            assert a[i] == i + 1
        a[()] = range(5)
        for i in xrange(5):
            assert a[i] == i

    def test_setslice_array(self):
        from numpypy import array
        a = array(range(5))
        b = array(range(2))
        a[1:4:2] = b
        assert a[1] == 0.
        assert a[3] == 1.
        b[::-1] = b
        assert b[0] == 0.
        assert b[1] == 0.

    def test_setslice_of_slice_array(self):
        from numpypy import array, zeros
        a = zeros(5)
        a[::2] = array([9., 10., 11.])
        assert a[0] == 9.
        assert a[2] == 10.
        assert a[4] == 11.
        a[1:4:2][::-1] = array([1., 2.])
        assert a[0] == 9.
        assert a[1] == 2.
        assert a[2] == 10.
        assert a[3] == 1.
        assert a[4] == 11.
        a = zeros(10)
        a[::2][::-1][::2] = array(range(1, 4))
        assert a[8] == 1.
        assert a[4] == 2.
        assert a[0] == 3.

    def test_setslice_list(self):
        from numpypy import array
        a = array(range(5), float)
        b = [0., 1.]
        a[1:4:2] = b
        assert a[1] == 0.
        assert a[3] == 1.

    def test_setslice_constant(self):
        from numpypy import array
        a = array(range(5), float)
        a[1:4:2] = 0.
        assert a[1] == 0.
        assert a[3] == 0.

    def test_scalar(self):
        from numpypy import array, dtype
        a = array(3)
        #assert a[0] == 3
        raises(IndexError, "a[0]")
        assert a.size == 1
        assert a.shape == ()
        assert a.dtype is dtype(int)

    def test_len(self):
        from numpypy import array
        a = array(range(5))
        assert len(a) == 5
        assert len(a + a) == 5

    def test_shape(self):
        from numpypy import array
        a = array(range(5))
        assert a.shape == (5,)
        b = a + a
        assert b.shape == (5,)
        c = a[:3]
        assert c.shape == (3,)

    def test_add(self):
        from numpypy import array
        a = array(range(5))
        b = a + a
        for i in range(5):
            assert b[i] == i + i

        a = array([True, False, True, False], dtype="?")
        b = array([True, True, False, False], dtype="?")
        c = a + b
        for i in range(4):
            assert c[i] == bool(a[i] + b[i])

    def test_add_other(self):
        from numpypy import array
        a = array(range(5))
        b = array([i for i in reversed(range(5))])
        c = a + b
        for i in range(5):
            assert c[i] == 4

    def test_add_constant(self):
        from numpypy import array
        a = array(range(5))
        b = a + 5
        for i in range(5):
            assert b[i] == i + 5

    def test_radd(self):
        from numpypy import array
        r = 3 + array(range(3))
        for i in range(3):
            assert r[i] == i + 3

    def test_add_list(self):
        from numpypy import array, ndarray
        a = array(range(5))
        b = list(reversed(range(5)))
        c = a + b
        assert isinstance(c, ndarray)
        for i in range(5):
            assert c[i] == 4

    def test_subtract(self):
        from numpypy import array
        a = array(range(5))
        b = a - a
        for i in range(5):
            assert b[i] == 0

    def test_subtract_other(self):
        from numpypy import array
        a = array(range(5))
        b = array([1, 1, 1, 1, 1])
        c = a - b
        for i in range(5):
            assert c[i] == i - 1

    def test_subtract_constant(self):
        from numpypy import array
        a = array(range(5))
        b = a - 5
        for i in range(5):
            assert b[i] == i - 5

    def test_mul(self):
        import numpypy

        a = numpypy.array(range(5))
        b = a * a
        for i in range(5):
            assert b[i] == i * i

        a = numpypy.array(range(5), dtype=bool)
        b = a * a
        assert b.dtype is numpypy.dtype(bool)
        assert b[0] is numpypy.False_
        for i in range(1, 5):
            assert b[i] is numpypy.True_

    def test_mul_constant(self):
        from numpypy import array
        a = array(range(5))
        b = a * 5
        for i in range(5):
            assert b[i] == i * 5

    def test_div(self):
        from math import isnan
        from numpypy import array, dtype, inf

        a = array(range(1, 6))
        b = a / a
        for i in range(5):
            assert b[i] == 1

        a = array(range(1, 6), dtype=bool)
        b = a / a
        assert b.dtype is dtype("int8")
        for i in range(5):
            assert b[i] == 1

        a = array([-1, 0, 1])
        b = array([0, 0, 0])
        c = a / b
        assert (c == [0, 0, 0]).all()

        a = array([-1.0, 0.0, 1.0])
        b = array([0.0, 0.0, 0.0])
        c = a / b
        assert c[0] == -inf
        assert isnan(c[1])
        assert c[2] == inf

        b = array([-0.0, -0.0, -0.0])
        c = a / b
        assert c[0] == inf
        assert isnan(c[1])
        assert c[2] == -inf

    def test_div_other(self):
        from numpypy import array
        a = array(range(5))
        b = array([2, 2, 2, 2, 2], float)
        c = a / b
        for i in range(5):
            assert c[i] == i / 2.0

    def test_div_constant(self):
        from numpypy import array
        a = array(range(5))
        b = a / 5.0
        for i in range(5):
            assert b[i] == i / 5.0

    def test_pow(self):
        from numpypy import array
        a = array(range(5), float)
        b = a ** a
        for i in range(5):
            assert b[i] == i ** i

        a = array(range(5))
        assert (a ** 2 == a * a).all()

    def test_pow_other(self):
        from numpypy import array
        a = array(range(5), float)
        b = array([2, 2, 2, 2, 2])
        c = a ** b
        for i in range(5):
            assert c[i] == i ** 2

    def test_pow_constant(self):
        from numpypy import array
        a = array(range(5), float)
        b = a ** 2
        for i in range(5):
            assert b[i] == i ** 2

    def test_mod(self):
        from numpypy import array
        a = array(range(1, 6))
        b = a % a
        for i in range(5):
            assert b[i] == 0

        a = array(range(1, 6), float)
        b = (a + 1) % a
        assert b[0] == 0
        for i in range(1, 5):
            assert b[i] == 1

    def test_mod_other(self):
        from numpypy import array
        a = array(range(5))
        b = array([2, 2, 2, 2, 2])
        c = a % b
        for i in range(5):
            assert c[i] == i % 2

    def test_mod_constant(self):
        from numpypy import array
        a = array(range(5))
        b = a % 2
        for i in range(5):
            assert b[i] == i % 2

    def test_pos(self):
        from numpypy import array
        a = array([1., -2., 3., -4., -5.])
        b = +a
        for i in range(5):
            assert b[i] == a[i]

        a = +array(range(5))
        for i in range(5):
            assert a[i] == i

    def test_neg(self):
        from numpypy import array
        a = array([1., -2., 3., -4., -5.])
        b = -a
        for i in range(5):
            assert b[i] == -a[i]

        a = -array(range(5), dtype="int8")
        for i in range(5):
            assert a[i] == -i

    def test_abs(self):
        from numpypy import array
        a = array([1., -2., 3., -4., -5.])
        b = abs(a)
        for i in range(5):
            assert b[i] == abs(a[i])

        a = abs(array(range(-5, 5), dtype="int8"))
        for i in range(-5, 5):
            assert a[i + 5] == abs(i)

    def test_auto_force(self):
        from numpypy import array
        a = array(range(5))
        b = a - 1
        a[2] = 3
        for i in range(5):
            assert b[i] == i - 1

        a = array(range(5))
        b = a + a
        c = b + b
        b[1] = 5
        assert c[1] == 4

    def test_getslice(self):
        from numpypy import array
        a = array(range(5))
        s = a[1:5]
        assert len(s) == 4
        for i in range(4):
            assert s[i] == a[i + 1]

        s = (a + a)[1:2]
        assert len(s) == 1
        assert s[0] == 2
        s[:1] = array([5])
        assert s[0] == 5

    def test_getslice_step(self):
        from numpypy import array
        a = array(range(10))
        s = a[1:9:2]
        assert len(s) == 4
        for i in range(4):
            assert s[i] == a[2 * i + 1]

    def test_slice_update(self):
        from numpypy import array
        a = array(range(5))
        s = a[0:3]
        s[1] = 10
        assert a[1] == 10
        a[2] = 20
        assert s[2] == 20

    def test_slice_invaidate(self):
        # check that slice shares invalidation list with
        from numpypy import array
        a = array(range(5))
        s = a[0:2]
        b = array([10, 11])
        c = s + b
        a[0] = 100
        assert c[0] == 10
        assert c[1] == 12
        d = s + b
        a[1] = 101
        assert d[0] == 110
        assert d[1] == 12

    def test_mean(self):
        from numpypy import array
        a = array(range(5))
        assert a.mean() == 2.0
        assert a[:4].mean() == 1.5

    def test_sum(self):
        from numpypy import array
        a = array(range(5))
        assert a.sum() == 10.0
        assert a[:4].sum() == 6.0

        a = array([True] * 5, bool)
        assert a.sum() == 5

    def test_prod(self):
        from numpypy import array
        a = array(range(1, 6))
        assert a.prod() == 120.0
        assert a[:4].prod() == 24.0

    def test_max(self):
        from numpypy import array
        a = array([-1.2, 3.4, 5.7, -3.0, 2.7])
        assert a.max() == 5.7
        b = array([])
        raises(ValueError, "b.max()")

    def test_max_add(self):
        from numpypy import array
        a = array([-1.2, 3.4, 5.7, -3.0, 2.7])
        assert (a + a).max() == 11.4

    def test_min(self):
        from numpypy import array
        a = array([-1.2, 3.4, 5.7, -3.0, 2.7])
        assert a.min() == -3.0
        b = array([])
        raises(ValueError, "b.min()")

    def test_argmax(self):
        from numpypy import array
        a = array([-1.2, 3.4, 5.7, -3.0, 2.7])
        r = a.argmax()
        assert r == 2
        b = array([])
        raises(ValueError, b.argmax)

        a = array(range(-5, 5))
        r = a.argmax()
        assert r == 9
        b = a[::2]
        r = b.argmax()
        assert r == 4
        r = (a + a).argmax()
        assert r == 9
        a = array([1, 0, 0])
        assert a.argmax() == 0
        a = array([0, 0, 1])
        assert a.argmax() == 2

    def test_argmin(self):
        from numpypy import array
        a = array([-1.2, 3.4, 5.7, -3.0, 2.7])
        assert a.argmin() == 3
        b = array([])
        raises(ValueError, "b.argmin()")

    def test_all(self):
        from numpypy import array
        a = array(range(5))
        assert a.all() == False
        a[0] = 3.0
        assert a.all() == True
        b = array([])
        assert b.all() == True

    def test_any(self):
        from numpypy import array, zeros
        a = array(range(5))
        assert a.any() == True
        b = zeros(5)
        assert b.any() == False
        c = array([])
        assert c.any() == False

    def test_dot(self):
        from numpypy import array, dot
        a = array(range(5))
        assert a.dot(a) == 30.0

        a = array(range(5))
        assert a.dot(range(5)) == 30
        assert dot(range(5), range(5)) == 30
        assert (dot(5, [1, 2, 3]) == [5, 10, 15]).all()

    def test_dot_constant(self):
        from numpypy import array
        a = array(range(5))
        b = a.dot(2.5)
        for i in xrange(5):
            assert b[i] == 2.5 * a[i]

    def test_dtype_guessing(self):
        from numpypy import array, dtype, float64, int8, bool_

        assert array([True]).dtype is dtype(bool)
        assert array([True, False]).dtype is dtype(bool)
        assert array([True, 1]).dtype is dtype(int)
        assert array([1, 2, 3]).dtype is dtype(int)
        assert array([1L, 2, 3]).dtype is dtype(long)
        assert array([1.2, True]).dtype is dtype(float)
        assert array([1.2, 5]).dtype is dtype(float)
        assert array([]).dtype is dtype(float)
        assert array([float64(2)]).dtype is dtype(float)
        assert array([int8(3)]).dtype is dtype("int8")
        assert array([bool_(True)]).dtype is dtype(bool)
        assert array([bool_(True), 3.0]).dtype is dtype(float)
        assert array([1 + 2j]).dtype is dtype(complex)
        assert array([1, 1 + 2j]).dtype is dtype(complex)

    def test_comparison(self):
        import operator
        from numpypy import array, dtype

        a = array(range(5))
        b = array(range(5), float)
        for func in [
            operator.eq, operator.ne, operator.lt, operator.le, operator.gt,
            operator.ge
        ]:
            c = func(a, 3)
            assert c.dtype is dtype(bool)
            for i in xrange(5):
                assert c[i] == func(a[i], 3)

            c = func(b, 3)
            assert c.dtype is dtype(bool)
            for i in xrange(5):
                assert c[i] == func(b[i], 3)

    def test_nonzero(self):
        from numpypy import array

        a = array([1, 2])
        raises(ValueError, bool, a)
        raises(ValueError, bool, a == a)
        assert bool(array(1))
        assert not bool(array(0))
        assert bool(array([1]))
        assert not bool(array([0]))

    def test_complex_basic(self):
        from numpypy import array

        x = array([1, 2, 3], complex)
        assert x[0] == complex(1, 0)
        x[0] = 1 + 3j
        assert x[0] == 1 + 3j
        assert x[2].real == 3
        assert x[2].imag == 0

    def test_slice_assignment(self):
        from numpypy import array
        a = array(range(5))
        a[::-1] = a
        assert (a == [0, 1, 2, 1, 0]).all()
        # but we force intermediates
        a = array(range(5))
        a[::-1] = a + a
        assert (a == [8, 6, 4, 2, 0]).all()

    def test_debug_repr(self):
        from numpypy import zeros, sin
        a = zeros(1)
        assert a.__debug_repr__() == 'Array'
        assert (a + a).__debug_repr__() == 'Call2(add, Array, Array)'
        assert (a[::2]).__debug_repr__() == 'Slice(Array)'
        assert (a + 2).__debug_repr__() == 'Call2(add, Array, Scalar)'
        assert (a + a.flat).__debug_repr__() == 'Call2(add, Array, FlatIter(Array))'
        assert sin(a).__debug_repr__() == 'Call1(sin, Array)'
        b = a + a
        b[0] = 3
        assert b.__debug_repr__() == 'Call2(add, forced=Array)'

class AppTestMultiDim(BaseNumpyAppTest):
    def test_init(self):
        import numpypy
        a = numpypy.zeros((2, 2))
        assert len(a) == 2

    def test_shape(self):
        import numpypy
        assert numpypy.zeros(1).shape == (1,)
        assert numpypy.zeros((2, 2)).shape == (2, 2)
        assert numpypy.zeros((3, 1, 2)).shape == (3, 1, 2)
        assert numpypy.array([[1], [2], [3]]).shape == (3, 1)
        assert len(numpypy.zeros((3, 1, 2))) == 3
        raises(TypeError, len, numpypy.zeros(()))
        raises(ValueError, numpypy.array, [[1, 2], 3])

    def test_getsetitem(self):
        import numpypy
        a = numpypy.zeros((2, 3, 1))
        raises(IndexError, a.__getitem__, (2, 0, 0))
        raises(IndexError, a.__getitem__, (0, 3, 0))
        raises(IndexError, a.__getitem__, (0, 0, 1))
        assert a[1, 1, 0] == 0
        a[1, 2, 0] = 3
        assert a[1, 2, 0] == 3
        assert a[1, 1, 0] == 0
        assert a[1, -1, 0] == 3

    def test_slices(self):
        import numpypy
        a = numpypy.zeros((4, 3, 2))
        raises(IndexError, a.__getitem__, (4,))
        raises(IndexError, a.__getitem__, (3, 3))
        raises(IndexError, a.__getitem__, (slice(None), 3))
        a[0, 1, 1] = 13
        a[1, 2, 1] = 15
        b = a[0]
        assert len(b) == 3
        assert b.shape == (3, 2)
        assert b[1, 1] == 13
        b = a[1]
        assert b.shape == (3, 2)
        assert b[2, 1] == 15
        b = a[:, 1]
        assert b.shape == (4, 2)
        assert b[0, 1] == 13
        b = a[:, 1, :]
        assert b.shape == (4, 2)
        assert b[0, 1] == 13
        b = a[1, 2]
        assert b[1] == 15
        b = a[:]
        assert b.shape == (4, 3, 2)
        assert b[1, 2, 1] == 15
        assert b[0, 1, 1] == 13
        b = a[:][:, 1][:]
        assert b[2, 1] == 0.0
        assert b[0, 1] == 13
        raises(IndexError, b.__getitem__, (4, 1))
        assert a[0][1][1] == 13
        assert a[1][2][1] == 15

    def test_init_2(self):
        import numpypy
        raises(ValueError, numpypy.array, [[1], 2])
        raises(ValueError, numpypy.array, [[1, 2], [3]])
        raises(ValueError, numpypy.array, [[[1, 2], [3, 4], 5]])
        raises(ValueError, numpypy.array, [[[1, 2], [3, 4], [5]]])
        a = numpypy.array([[1, 2], [4, 5]])
        assert a[0, 1] == 2
        assert a[0][1] == 2
        a = numpypy.array(([[[1, 2], [3, 4], [5, 6]]]))
        assert (a[0, 1] == [3, 4]).all()

    def test_setitem_slice(self):
        import numpypy
        a = numpypy.zeros((3, 4))
        a[1] = [1, 2, 3, 4]
        assert a[1, 2] == 3
        raises(TypeError, a[1].__setitem__, [1, 2, 3])
        a = numpypy.array([[1, 2], [3, 4]])
        assert (a == [[1, 2], [3, 4]]).all()
        a[1] = numpypy.array([5, 6])
        assert (a == [[1, 2], [5, 6]]).all()
        a[:, 1] = numpypy.array([8, 10])
        assert (a == [[1, 8], [5, 10]]).all()
        a[0, :: -1] = numpypy.array([11, 12])
        assert (a == [[12, 11], [5, 10]]).all()

    def test_ufunc(self):
        from numpypy import array
        a = array([[1, 2], [3, 4], [5, 6]])
        assert ((a + a) == \
            array([[1 + 1, 2 + 2], [3 + 3, 4 + 4], [5 + 5, 6 + 6]])).all()

    def test_getitem_add(self):
        from numpypy import array
        a = array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        assert (a + a)[1, 1] == 8

    def test_ufunc_negative(self):
        from numpypy import array, negative
        a = array([[1, 2], [3, 4]])
        b = negative(a + a)
        assert (b == [[-2, -4], [-6, -8]]).all()

    def test_getitem_3(self):
        from numpypy import array
        a = array([[1, 2], [3, 4], [5, 6], [7, 8],
                   [9, 10], [11, 12], [13, 14]])
        b = a[::2]
        print a
        print b
        assert (b == [[1, 2], [5, 6], [9, 10], [13, 14]]).all()
        c = b + b
        assert c[1][1] == 12

    def test_multidim_ones(self):
        from numpypy import ones
        a = ones((1, 2, 3))
        assert a[0, 1, 2] == 1.0

    def test_broadcast_ufunc(self):
        from numpypy import array
        a = array([[1, 2], [3, 4], [5, 6]])
        b = array([5, 6])
        c = ((a + b) == [[1 + 5, 2 + 6], [3 + 5, 4 + 6], [5 + 5, 6 + 6]])
        assert c.all()

    def test_broadcast_setslice(self):
        from numpypy import zeros, ones
        a = zeros((100, 100))
        b = ones(100)
        a[:, :] = b
        assert a[13, 15] == 1

    def test_broadcast_shape_agreement(self):
        from numpypy import zeros, array
        a = zeros((3, 1, 3))
        b = array(((10, 11, 12), (20, 21, 22), (30, 31, 32)))
        c = ((a + b) == [b, b, b])
        assert c.all()
        a = array((((10, 11, 12), ), ((20, 21, 22), ), ((30, 31, 32), )))
        assert(a.shape == (3, 1, 3))
        d = zeros((3, 3))
        c = ((a + d) == [b, b, b])
        c = ((a + d) == array([[[10., 11., 12.]] * 3,
                               [[20., 21., 22.]] * 3, [[30., 31., 32.]] * 3]))
        assert c.all()

    def test_broadcast_scalar(self):
        from numpypy import zeros
        a = zeros((4, 5), 'd')
        a[:, 1] = 3
        assert a[2, 1] == 3
        assert a[0, 2] == 0
        a[0, :] = 5
        assert a[0, 3] == 5
        assert a[2, 1] == 3
        assert a[3, 2] == 0

    def test_broadcast_call2(self):
        from numpypy import zeros, ones
        a = zeros((4, 1, 5))
        b = ones((4, 3, 5))
        b[:] = (a + a)
        assert (b == zeros((4, 3, 5))).all()

    def test_argmax(self):
        from numpypy import array
        a = array([[1, 2], [3, 4], [5, 6]])
        assert a.argmax() == 5
        assert a[:2, ].argmax() == 3

    def test_broadcast_wrong_shapes(self):
        from numpypy import zeros
        a = zeros((4, 3, 2))
        b = zeros((4, 2))
        exc = raises(ValueError, lambda: a + b)
        assert str(exc.value) == "operands could not be broadcast" \
            " together with shapes (4,3,2) (4,2)"

    def test_reduce(self):
        from numpypy import array
        a = array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        assert a.sum() == (13 * 12) / 2
        b = a[1:, 1::2]
        c = b + b
        assert c.sum() == (6 + 8 + 10 + 12) * 2

    def test_transpose(self):
        from numpypy import array
        a = array(((range(3), range(3, 6)),
                   (range(6, 9), range(9, 12)),
                   (range(12, 15), range(15, 18)),
                   (range(18, 21), range(21, 24))))
        assert a.shape == (4, 2, 3)
        b = a.T
        assert b.shape == (3, 2, 4)
        assert(b[0, :, 0] == [0, 3]).all()
        b[:, 0, 0] = 1000
        assert(a[0, 0, :] == [1000, 1000, 1000]).all()
        a = array(range(5))
        b = a.T
        assert(b == range(5)).all()
        a = array((range(10), range(20, 30)))
        b = a.T
        assert(b[:, 0] == a[0, :]).all()

    def test_flatiter(self):
        from numpypy import array, flatiter
        a = array([[10, 30], [40, 60]])
        f_iter = a.flat
        assert f_iter.next() == 10
        assert f_iter.next() == 30
        assert f_iter.next() == 40
        assert f_iter.next() == 60
        raises(StopIteration, "f_iter.next()")
        raises(TypeError, "flatiter()")
        s = 0
        for k in a.flat:
            s += k
        assert s == 140

    def test_flatiter_array_conv(self):
        from numpypy import array, dot
        a = array([1, 2, 3])
        assert dot(a.flat, a.flat) == 14

    def test_slice_copy(self):
        from numpypy import zeros
        a = zeros((10, 10))
        b = a[0].copy()
        assert (b == zeros(10)).all()

class AppTestSupport(BaseNumpyAppTest):
    def setup_class(cls):
        import struct
        BaseNumpyAppTest.setup_class.im_func(cls)
        cls.w_data = cls.space.wrap(struct.pack('dddd', 1, 2, 3, 4))

    def test_fromstring(self):
        from numpypy import fromstring
        a = fromstring(self.data)
        for i in range(4):
            assert a[i] == i + 1
        raises(ValueError, fromstring, "abc")


class AppTestRepr(BaseNumpyAppTest):
    def test_repr(self):
        from numpypy import array, zeros
        a = array(range(5), float)
        assert repr(a) == "array([0.0, 1.0, 2.0, 3.0, 4.0])"
        a = array([], float)
        assert repr(a) == "array([], dtype=float64)"
        a = zeros(1001)
        assert repr(a) == "array([0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0])"
        a = array(range(5), long)
        assert repr(a) == "array([0, 1, 2, 3, 4])"
        a = array([], long)
        assert repr(a) == "array([], dtype=int64)"
        a = array([True, False, True, False], "?")
        assert repr(a) == "array([True, False, True, False], dtype=bool)"

    def test_repr_multi(self):
        from numpypy import array, zeros
        a = zeros((3, 4))
        assert repr(a) == '''array([[0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0]])'''
        a = zeros((2, 3, 4))
        assert repr(a) == '''array([[[0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]],

       [[0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]]])'''

    def test_repr_slice(self):
        from numpypy import array, zeros
        a = array(range(5), float)
        b = a[1::2]
        assert repr(b) == "array([1.0, 3.0])"
        a = zeros(2002)
        b = a[::2]
        assert repr(b) == "array([0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0])"
        a = array((range(5), range(5, 10)), dtype="int16")
        b = a[1, 2:]
        assert repr(b) == "array([7, 8, 9], dtype=int16)"
        # an empty slice prints its shape
        b = a[2:1, ]
        assert repr(b) == "array([], shape=(0, 5), dtype=int16)"

    def test_str(self):
        from numpypy import array, zeros
        a = array(range(5), float)
        assert str(a) == "[0.0 1.0 2.0 3.0 4.0]"
        assert str((2 * a)[:]) == "[0.0 2.0 4.0 6.0 8.0]"
        a = zeros(1001)
        assert str(a) == "[0.0 0.0 0.0 ..., 0.0 0.0 0.0]"

        a = array(range(5), dtype=long)
        assert str(a) == "[0 1 2 3 4]"
        a = array([True, False, True, False], dtype="?")
        assert str(a) == "[True False True False]"

        a = array(range(5), dtype="int8")
        assert str(a) == "[0 1 2 3 4]"

        a = array(range(5), dtype="int16")
        assert str(a) == "[0 1 2 3 4]"

        a = array((range(5), range(5, 10)), dtype="int16")
        assert str(a) == "[[0 1 2 3 4]\n [5 6 7 8 9]]"

        a = array(3, dtype=int)
        assert str(a) == "3"

        a = zeros((400, 400), dtype=int)
        assert str(a) == "[[0 0 0 ..., 0 0 0]\n [0 0 0 ..., 0 0 0]\n" \
           " [0 0 0 ..., 0 0 0]\n ..., \n [0 0 0 ..., 0 0 0]\n" \
           " [0 0 0 ..., 0 0 0]\n [0 0 0 ..., 0 0 0]]"
        a = zeros((2, 2, 2))
        r = str(a)
        assert r == '[[[0.0 0.0]\n  [0.0 0.0]]\n\n [[0.0 0.0]\n  [0.0 0.0]]]'

    def test_str_slice(self):
        from numpypy import array, zeros
        a = array(range(5), float)
        b = a[1::2]
        assert str(b) == "[1.0 3.0]"
        a = zeros(2002)
        b = a[::2]
        assert str(b) == "[0.0 0.0 0.0 ..., 0.0 0.0 0.0]"
        a = array((range(5), range(5, 10)), dtype="int16")
        b = a[1, 2:]
        assert str(b) == "[7 8 9]"
        b = a[2:1, ]
        assert str(b) == "[]"


class AppTestRanges(BaseNumpyAppTest):
    def test_arange(self):
        from numpypy import arange, array, dtype
        a = arange(3)
        assert (a == [0, 1, 2]).all()
        assert a.dtype is dtype(int)
        a = arange(3.0)
        assert (a == [0., 1., 2.]).all()
        assert a.dtype is dtype(float)
        a = arange(3, 7)
        assert (a == [3, 4, 5, 6]).all()
        assert a.dtype is dtype(int)
        a = arange(3, 7, 2)
        assert (a == [3, 5]).all()
        a = arange(3, dtype=float)
        assert (a == [0., 1., 2.]).all()
        assert a.dtype is dtype(float)
        a = arange(0, 0.8, 0.1)
        assert len(a) == 8
        assert arange(False, True, True).dtype is dtype(int)
