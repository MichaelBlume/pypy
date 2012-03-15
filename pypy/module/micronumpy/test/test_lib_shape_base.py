from pypy.module.micronumpy.test.test_base import BaseNumpyAppTest

def assert_equal(a,b):
    result = a == b
    if isinstance(result, bool):
        assert result
    else:
        assert result.all()

class TestApplyAlongAxis(BaseNumpyAppTest):
    def test_simple(self):
        from numpy import ones, array_equal, shape
        from numpy.lib.shape_base import apply_along_axis
        a = ones((20,10),'d')
        assert array_equal(apply_along_axis(len,0,a),len(a)*ones(shape(a)[1]))

    def test_simple101(self,level=11):
        from numpy import ones, array_equal, shape
        from numpy.lib.shape_base import apply_along_axis
        a = ones((10,101),'d')
        assert array_equal(apply_along_axis(len,0,a),len(a)*ones(shape(a)[1]))

    def test_3d(self):
        from numpy import array_equal, arange
        from numpy.lib.shape_base import apply_along_axis
        a = arange(27).reshape((3,3,3))
        assert array_equal(apply_along_axis(sum,0,a),
                           [[27,30,33],[36,39,42],[45,48,51]])


class TestApplyOverAxes(BaseNumpyAppTest):
    def test_simple(self):
        '''Had to change the result of this test, which is weird. Got the same
           result with stock NumPy though.'''
        from numpy import array_equal, arange, array, newaxis
        from numpy.lib.shape_base import apply_over_axes
        a = arange(24).reshape(2,3,4)
        aoa_a = apply_over_axes(sum, a, [0,2])
        expected =array([[14, 16, 18, 20],
                         [22, 24, 26, 28],
                         [30, 32, 34, 36]])[:,:,newaxis]
        assert array_equal(aoa_a, expected)


class TestArraySplit(BaseNumpyAppTest):
    def test_integer_0_split(self):
        from numpy import arange
        from numpy.lib.shape_base import array_split
        a = arange(10)
        try:
            res = array_split(a,0)
            assert (0) # it should have thrown a value error
        except ValueError:
            pass

    def test_integer_split(self):
        from numpy import arange, array
        from numpy.lib.shape_base import array_split
        a = arange(10)
        res = array_split(a,1)
        desired = [arange(10)]
        compare_results(res,desired)

        res = array_split(a,2)
        desired = [arange(5),arange(5,10)]
        compare_results(res,desired)

        res = array_split(a,3)
        desired = [arange(4),arange(4,7),arange(7,10)]
        compare_results(res,desired)

        res = array_split(a,4)
        desired = [arange(3),arange(3,6),arange(6,8),arange(8,10)]
        compare_results(res,desired)

        res = array_split(a,5)
        desired = [arange(2),arange(2,4),arange(4,6),arange(6,8),arange(8,10)]
        compare_results(res,desired)

        res = array_split(a,6)
        desired = [arange(2),arange(2,4),arange(4,6),arange(6,8),arange(8,9),
                   arange(9,10)]
        compare_results(res,desired)

        res = array_split(a,7)
        desired = [arange(2),arange(2,4),arange(4,6),arange(6,7),arange(7,8),
                   arange(8,9), arange(9,10)]
        compare_results(res,desired)

        res = array_split(a,8)
        desired = [arange(2),arange(2,4),arange(4,5),arange(5,6),arange(6,7),
                   arange(7,8), arange(8,9), arange(9,10)]
        compare_results(res,desired)

        res = array_split(a,9)
        desired = [arange(2),arange(2,3),arange(3,4),arange(4,5),arange(5,6),
                   arange(6,7), arange(7,8), arange(8,9), arange(9,10)]
        compare_results(res,desired)

        res = array_split(a,10)
        desired = [arange(1),arange(1,2),arange(2,3),arange(3,4),
                   arange(4,5),arange(5,6), arange(6,7), arange(7,8),
                   arange(8,9), arange(9,10)]
        compare_results(res,desired)

        res = array_split(a,11)
        desired = [arange(1),arange(1,2),arange(2,3),arange(3,4),
                   arange(4,5),arange(5,6), arange(6,7), arange(7,8),
                   arange(8,9), arange(9,10),array([])]
        compare_results(res,desired)

    def test_integer_split_2D_rows(self):
        from numpy import arange, array
        from numpy.lib.shape_base import array_split
        a = array([arange(10),arange(10)])
        res = array_split(a,3,axis=0)
        desired = [array([arange(10)]),array([arange(10)]),array([])]
        compare_results(res,desired)

    def test_integer_split_2D_cols(self):
        from numpy import arange, array
        from numpy.lib.shape_base import array_split
        a = array([arange(10),arange(10)])
        res = array_split(a,3,axis=-1)
        desired = [array([arange(4),arange(4)]),
                   array([arange(4,7),arange(4,7)]),
                   array([arange(7,10),arange(7,10)])]
        compare_results(res,desired)

    def test_integer_split_2D_default(self):
        """ This will fail if we change default axis
        """
        from numpy import arange, array
        from numpy.lib.shape_base import array_split
        a = array([arange(10),arange(10)])
        res = array_split(a,3)
        desired = [array([arange(10)]),array([arange(10)]),array([])]
        compare_results(res,desired)
    #perhaps should check higher dimensions

    def test_index_split_simple(self):
        from numpy import arange, array
        from numpy.lib.shape_base import array_split
        a = arange(10)
        indices = [1,5,7]
        res = array_split(a,indices,axis=-1)
        desired = [arange(0,1),arange(1,5),arange(5,7),arange(7,10)]
        compare_results(res,desired)

    def test_index_split_low_bound(self):
        from numpy import arange, array
        from numpy.lib.shape_base import array_split
        a = arange(10)
        indices = [0,5,7]
        res = array_split(a,indices,axis=-1)
        desired = [array([]),arange(0,5),arange(5,7),arange(7,10)]
        compare_results(res,desired)

    def test_index_split_high_bound(self):
        from numpy import arange, array
        from numpy.lib.shape_base import array_split
        a = arange(10)
        indices = [0,5,7,10,12]
        res = array_split(a,indices,axis=-1)
        desired = [array([]),arange(0,5),arange(5,7),arange(7,10),
                   array([]),array([])]
        compare_results(res,desired)


class TestSplit(BaseNumpyAppTest):
    """* This function is essentially the same as array_split,
         except that it test if splitting will result in an
         equal split.  Only test for this case.
    *"""
    def test_equal_split(self):
        from numpy import arange
        from numpy.lib.shape_base import split
        a = arange(10)
        res = split(a,2)
        desired = [arange(5),arange(5,10)]
        compare_results(res,desired)

    def test_unequal_split(self):
        from numpy import arange
        from numpy.lib.shape_base import split
        a = arange(10)
        try:
            res = split(a,3)
            assert (0) # should raise an error
        except ValueError:
            pass


class TestDstack(BaseNumpyAppTest):
    def test_0D_array(self):
        from numpy import array, array_equal
        from numpy.lib.shape_base import dstack
        a = array(1); b = array(2);
        res=dstack([a,b])
        desired = array([[[1,2]]])
        assert array_equal(res,desired)

    def test_1D_array(self):
        from numpy import array, array_equal
        from numpy.lib.shape_base import dstack
        a = array([1]); b = array([2]);
        res=dstack([a,b])
        desired = array([[[1,2]]])
        assert array_equal(res,desired)

    def test_2D_array(self):
        from numpy import array, array_equal
        from numpy.lib.shape_base import dstack
        a = array([[1],[2]]); b = array([[1],[2]]);
        res=dstack([a,b])
        desired = array([[[1,1]],[[2,2,]]])
        assert array_equal(res,desired)

    def test_2D_array2(self):
        from numpy import array, array_equal
        from numpy.lib.shape_base import dstack
        a = array([1,2]); b = array([1,2]);
        res=dstack([a,b])
        desired = array([[[1,1],[2,2]]])
        assert array_equal(res,desired)

""" array_split has more comprehensive test of splitting.
    only do simple test on hsplit, vsplit, and dsplit
"""
class TestHsplit(BaseNumpyAppTest):
    """ only testing for integer splits.
    """
    def test_0D_array(self):
        from numpy import array
        from numpy.lib.shape_base import hsplit
        a= array(1)
        try:
            hsplit(a,2)
            assert (0)
        except ValueError:
            pass

    def test_1D_array(self):
        from numpy import array
        from numpy.lib.shape_base import hsplit
        a= array([1,2,3,4])
        res = hsplit(a,2)
        desired = [array([1,2]),array([3,4])]
        compare_results(res,desired)

    def test_2D_array(self):
        from numpy import array
        from numpy.lib.shape_base import hsplit
        a= array([[1,2,3,4],
                  [1,2,3,4]])
        res = hsplit(a,2)
        desired = [array([[1,2],[1,2]]),array([[3,4],[3,4]])]
        compare_results(res,desired)


class TestVsplit(BaseNumpyAppTest):
    """ only testing for integer splits.
    """
    def test_1D_array(self):
        from numpy import array
        from numpy.lib.shape_base import vsplit
        a= array([1,2,3,4])
        try:
            vsplit(a,2)
            assert (0)
        except ValueError:
            pass

    def test_2D_array(self):
        from numpy import array
        from numpy.lib.shape_base import vsplit
        a= array([[1,2,3,4],
                  [1,2,3,4]])
        res = vsplit(a,2)
        desired = [array([[1,2,3,4]]),array([[1,2,3,4]])]
        compare_results(res,desired)


class TestDsplit(BaseNumpyAppTest):
    """ only testing for integer splits.
    """
    def test_2D_array(self):
        from numpy import array
        from numpy.lib.shape_base import dsplit
        a= array([[1,2,3,4],
                  [1,2,3,4]])
        try:
            dsplit(a,2)
            assert (0)
        except ValueError:
            pass

    def test_3D_array(self):
        from numpy import array
        from numpy.lib.shape_base import dsplit
        a= array([[[1,2,3,4],
                   [1,2,3,4]],
                  [[1,2,3,4],
                   [1,2,3,4]]])
        res = dsplit(a,2)
        desired = [array([[[1,2],[1,2]],[[1,2],[1,2]]]),
                   array([[[3,4],[3,4]],[[3,4],[3,4]]])]
        compare_results(res,desired)


class TestSqueeze(BaseNumpyAppTest):

    @staticmethod
    def rand(*args):
        """Returns an array of random numbers with the given shape.

        This only uses the standard library, so it is useful for testing purposes.
        """
        import random
        from numpy.core import zeros, float64
        results = zeros(args, float64)
        f = results.flat
        for i in range(len(f)):
            f[i] = random.random()
        return results

    def test_basic(self):
        from numpy import reshape, array_equal, ndarray
        from numpy.core.fromnumeric import squeeze
        a = self.rand(20,10,10,1,1)
        b = self.rand(20,1,10,1,20)
        c = self.rand(1,1,20,10)
        assert array_equal(squeeze(a),reshape(a,(20,10,10)))
        assert array_equal(squeeze(b),reshape(b,(20,10,20)))
        assert array_equal(squeeze(c),reshape(c,(20,10)))

        # Squeezing to 0-dim should still give an ndarray
        a = [[[1.5]]]
        res = squeeze(a)
        assert_equal(res, 1.5)
        assert_equal(res.ndim, 0)
        assert_equal(type(res), ndarray)


class TestKron(BaseNumpyAppTest):
    def test_return_type(self):
        from numpy import ndarray, matrix, ones, asmatrix
        from numpy.lib.shape_base import kron
        a = ones([2,2])
        m = asmatrix(a)
        assert_equal(type(kron(a,a)), ndarray)
        assert_equal(type(kron(m,m)), matrix)
        assert_equal(type(kron(a,m)), matrix)
        assert_equal(type(kron(m,a)), matrix)
        class myarray(ndarray):
            __array_priority__ = 0.0
        ma = myarray(a.shape, a.dtype, a.data)
        assert_equal(type(kron(a,a)), ndarray)
        assert_equal(type(kron(ma,ma)), myarray)
        assert_equal(type(kron(a,ma)), ndarray)
        assert_equal(type(kron(ma,a)), myarray)


class TestTile(BaseNumpyAppTest):
    def test_basic(self):
        from numpy import array
        from numpy.lib.shape_base import tile
        a = array([0,1,2])
        b = [[1,2],[3,4]]
        assert_equal(tile(a,2), [0,1,2,0,1,2])
        assert_equal(tile(a,(2,2)), [[0,1,2,0,1,2],[0,1,2,0,1,2]])
        assert_equal(tile(a,(1,2)), [[0,1,2,0,1,2]])
        assert_equal(tile(b, 2), [[1,2,1,2],[3,4,3,4]])
        assert_equal(tile(b,(2,1)),[[1,2],[3,4],[1,2],[3,4]])
        assert_equal(tile(b,(2,2)),[[1,2,1,2],[3,4,3,4],
                                    [1,2,1,2],[3,4,3,4]])

    def test_empty(self):
        from numpy import array
        from numpy.lib.shape_base import tile
        a = array([[[]]])
        d = tile(a,(3,2,5)).shape
        assert_equal(d,(3,2,0))

    def test_kroncompare(self):
        import numpy.random as nr
        from numpy import ones
        from numpy.lib.shape_base import tile, kron
        reps=[(2,),(1,2),(2,1),(2,2),(2,3,2),(3,2)]
        shape=[(3,),(2,3),(3,4,3),(3,2,3),(4,3,2,4),(2,2)]
        for s in shape:
            b = nr.randint(0,10,size=s)
            for r in reps:
                a = ones(r, b.dtype)
                large = tile(b, r)
                klarge = kron(a, b)
                assert_equal(large, klarge)


# Utility
def compare_results(res,desired):
    from numpy import array_equal
    for i in range(len(desired)):
        assert array_equal(res[i],desired[i])
