from pypy.interpreter.baseobjspace import ObjSpace, W_Root, Wrappable
from pypy.interpreter.error import OperationError, operationerrfmt
from pypy.interpreter.gateway import interp2app, unwrap_spec
from pypy.interpreter.typedef import TypeDef, GetSetProperty
from pypy.rlib import jit
from pypy.rpython.lltypesystem import lltype
from pypy.tool.sourcetools import func_with_new_name
import math

def dummy1(v):
    assert isinstance(v, float)
    return v

def dummy2(v):
    assert isinstance(v, float)
    return v

TP = lltype.Array(lltype.Float, hints={'nolength': True})

numpy_driver = jit.JitDriver(greens = ['signature'],
                             reds = ['result_size', 'i', 'self', 'result'])
all_driver = jit.JitDriver(greens=['signature'], reds=['i', 'size', 'self'])
any_driver = jit.JitDriver(greens=['signature'], reds=['i', 'size', 'self'])

class Signature(object):
    def __init__(self):
        self.transitions = {}

    def transition(self, target):
        if target in self.transitions:
            return self.transitions[target]
        self.transitions[target] = new = Signature()
        return new

def pos(v):
    return v
def neg(v):
    return -v
def absolute(v):
    return abs(v)
def add(v1, v2):
    return v1 + v2
def sub(v1, v2):
    return v1 - v2
def mul(v1, v2):
    return v1 * v2
def div(v1, v2):
    return v1 / v2
def power(v1, v2):
    return math.pow(v1, v2)
def mod(v1, v2):
    return math.fmod(v1, v2)
def maximum(v1, v2):
    return max(v1, v2)
def minimum(v1, v2):
    return min(v1, v2)

class BaseArray(Wrappable):
    def __init__(self):
        self.invalidates = []

    def invalidated(self):
        if self.invalidates:
            self._invalidated()

    def _invalidated(self):
        for arr in self.invalidates:
            arr.force_if_needed()
        del self.invalidates[:]

    def _unop_impl(function):
        signature = Signature()
        def impl(self, space):
            new_sig = self.signature.transition(signature)
            res = Call1(
                function,
                self,
                new_sig)
            self.invalidates.append(res)
            return space.wrap(res)
        return func_with_new_name(impl, "uniop_%s_impl" % function.__name__)

    descr_pos = _unop_impl(pos)
    descr_neg = _unop_impl(neg)
    descr_abs = _unop_impl(absolute)

    def _binop_impl(function):
        signature = Signature()
        def impl(self, space, w_other):
            w_other = convert_to_array(space, w_other)
            new_sig = self.signature.transition(signature)
            res = Call2(
                function,
                self,
                w_other,
                new_sig.transition(w_other.signature)
            )
            w_other.invalidates.append(res)
            self.invalidates.append(res)
            return space.wrap(res)
        return func_with_new_name(impl, "binop_%s_impl" % function.__name__)

    descr_add = _binop_impl(add)
    descr_sub = _binop_impl(sub)
    descr_mul = _binop_impl(mul)
    descr_div = _binop_impl(div)
    descr_pow = _binop_impl(power)
    descr_mod = _binop_impl(mod)

    def _binop_right_impl(function):
        signature = Signature()
        def impl(self, space, w_other):
            new_sig = self.signature.transition(signature)
            w_other = FloatWrapper(space.float_w(w_other))
            res = Call2(
                function,
                w_other,
                self,
                new_sig.transition(w_other.signature)
            )
            self.invalidates.append(res)
            return space.wrap(res)
        return func_with_new_name(impl,
                                  "binop_right_%s_impl" % function.__name__)

    descr_radd = _binop_right_impl(add)
    descr_rsub = _binop_right_impl(sub)
    descr_rmul = _binop_right_impl(mul)
    descr_rdiv = _binop_right_impl(div)
    descr_rpow = _binop_right_impl(power)
    descr_rmod = _binop_right_impl(mod)

    def _reduce_sum_prod_impl(function, init):
        reduce_driver = jit.JitDriver(greens=['signature'],
                         reds = ['i', 'size', 'self', 'result'])

        def loop(self, result, size):
            i = 0
            while i < size:
                reduce_driver.jit_merge_point(signature=self.signature,
                                              self=self, size=size, i=i,
                                              result=result)
                result = function(result, self.eval(i))
                i += 1
            return result

        def impl(self, space):
            return space.wrap(loop(self, init, self.find_size()))
        return func_with_new_name(impl, "reduce_%s_impl" % function.__name__)

    def _reduce_max_min_impl(function):
        reduce_driver = jit.JitDriver(greens=['signature'],
                         reds = ['i', 'size', 'self', 'result'])
        def loop(self, result, size):
            i = 1
            while i < size:
                reduce_driver.jit_merge_point(signature=self.signature,
                                              self=self, size=size, i=i,
                                              result=result)
                result = function(result, self.eval(i))
                i += 1
            return result

        def impl(self, space):
            size = self.find_size()
            if size == 0:
                raise OperationError(space.w_ValueError,
                    space.wrap("Can't call %s on zero-size arrays" \
                            % function.__name__))
            return space.wrap(loop(self, self.eval(0), size))
        return func_with_new_name(impl, "reduce_%s_impl" % function.__name__)

    def _reduce_argmax_argmin_impl(function):
        reduce_driver = jit.JitDriver(greens=['signature'],
                         reds = ['i', 'size', 'result', 'self', 'cur_best'])
        def loop(self, size):
            result = 0
            cur_best = self.eval(0)
            i = 1
            while i < size:
                reduce_driver.jit_merge_point(signature=self.signature,
                                              self=self, size=size, i=i,
                                              result=result, cur_best=cur_best)
                new_best = function(cur_best, self.eval(i))
                if new_best != cur_best:
                    result = i
                    cur_best = new_best
                i += 1
            return result
        def impl(self, space):
            size = self.find_size()
            if size == 0:
                raise OperationError(space.w_ValueError,
                    space.wrap("Can't call %s on zero-size arrays" \
                            % function.__name__))
            return space.wrap(loop(self, size))
        return func_with_new_name(impl, "reduce_arg%s_impl" % function.__name__)

    def _all(self):
        size = self.find_size()
        i = 0
        while i < size:
            all_driver.jit_merge_point(signature=self.signature, self=self, size=size, i=i)
            if not self.eval(i):
                return False
            i += 1
        return True
    def descr_all(self, space):
        return space.wrap(self._all())

    def _any(self):
        size = self.find_size()
        i = 0
        while i < size:
            any_driver.jit_merge_point(signature=self.signature, self=self, size=size, i=i)
            if self.eval(i):
                return True
            i += 1
        return False
    def descr_any(self, space):
        return space.wrap(self._any())

    descr_sum = _reduce_sum_prod_impl(add, 0.0)
    descr_prod = _reduce_sum_prod_impl(mul, 1.0)
    descr_max = _reduce_max_min_impl(maximum)
    descr_min = _reduce_max_min_impl(minimum)
    descr_argmax = _reduce_argmax_argmin_impl(maximum)
    descr_argmin = _reduce_argmax_argmin_impl(minimum)

    def descr_dot(self, space, w_other):
        if isinstance(w_other, BaseArray):
            w_res = self.descr_mul(space, w_other)
            assert isinstance(w_res, BaseArray)
            return w_res.descr_sum(space)
        else:
            return self.descr_mul(space, w_other)

    def get_concrete(self):
        raise NotImplementedError

    def descr_get_shape(self, space):
        return space.newtuple([self.descr_len(space)])

    def descr_len(self, space):
        return self.get_concrete().descr_len(space)

    def descr_getitem(self, space, w_idx):
        # TODO: indexing by tuples
        start, stop, step, slice_length = space.decode_index4(w_idx, self.find_size())
        if step == 0:
            # Single index
            return space.wrap(self.get_concrete().getitem(start))
        else:
            # Slice
            res = SingleDimSlice(start, stop, step, slice_length, self, self.signature.transition(SingleDimSlice.static_signature))
            return space.wrap(res)

    @unwrap_spec(item=int, value=float)
    def descr_setitem(self, space, item, value):
        self.invalidated()
        return self.get_concrete().descr_setitem(space, item, value)

    def descr_mean(self, space):
        return space.wrap(space.float_w(self.descr_sum(space))/self.find_size())

def convert_to_array (space, w_obj):
    if isinstance(w_obj, BaseArray):
        return w_obj
    elif space.issequence_w(w_obj):
        # Convert to array.
        return new_numarray(space, w_obj)
    else:
        # If it's a scalar
        return FloatWrapper(space.float_w(w_obj))

class FloatWrapper(BaseArray):
    """
    Intermediate class representing a float literal.
    """
    _immutable_fields_ = ["float_value"]
    signature = Signature()

    def __init__(self, float_value):
        BaseArray.__init__(self)
        self.float_value = float_value

    def find_size(self):
        raise ValueError

    def eval(self, i):
        return self.float_value

class VirtualArray(BaseArray):
    """
    Class for representing virtual arrays, such as binary ops or ufuncs
    """
    def __init__(self, signature):
        BaseArray.__init__(self)
        self.forced_result = None
        self.signature = signature

    def _del_sources(self):
        # Function for deleting references to source arrays, to allow garbage-collecting them
        raise NotImplementedError

    def compute(self):
        i = 0
        signature = self.signature
        result_size = self.find_size()
        result = SingleDimArray(result_size)
        while i < result_size:
            numpy_driver.jit_merge_point(signature=signature,
                                         result_size=result_size, i=i,
                                         self=self, result=result)
            result.storage[i] = self.eval(i)
            i += 1
        return result

    def force_if_needed(self):
        if self.forced_result is None:
            self.forced_result = self.compute()
            self._del_sources()

    def get_concrete(self):
        self.force_if_needed()
        return self.forced_result

    def eval(self, i):
        if self.forced_result is not None:
            return self.forced_result.eval(i)
        return self._eval(i)

    def find_size(self):
        if self.forced_result is not None:
            # The result has been computed and sources may be unavailable
            return self.forced_result.find_size()
        return self._find_size()


class Call1(VirtualArray):
    _immutable_fields_ = ["function", "values"]

    def __init__(self, function, values, signature):
        VirtualArray.__init__(self, signature)
        self.function = function
        self.values = values

    def _del_sources(self):
        self.values = None

    def _find_size(self):
        return self.values.find_size()

    def _eval(self, i):
        return self.function(self.values.eval(i))

class Call2(VirtualArray):
    """
    Intermediate class for performing binary operations.
    """
    _immutable_fields_ = ["function", "left", "right"]

    def __init__(self, function, left, right, signature):
        VirtualArray.__init__(self, signature)
        self.function = function
        self.left = left
        self.right = right

    def _del_sources(self):
        self.left = None
        self.right = None

    def _find_size(self):
        try:
            return self.left.find_size()
        except ValueError:
            pass
        return self.right.find_size()

    def _eval(self, i):
        lhs, rhs = self.left.eval(i), self.right.eval(i)
        return self.function(lhs, rhs)

class ViewArray(BaseArray):
    """
    Class for representing views of arrays, they will reflect changes of parent
    arrays. Example: slices
    """
    _immutable_fields_ = ["parent"]

    def __init__(self, parent, signature):
        BaseArray.__init__(self)
        self.signature = signature
        self.parent = parent
        self.invalidates = parent.invalidates

    def get_concrete(self):
        # in fact, ViewArray never gets "concrete" as it never stores data.
        # This implementation is needed for BaseArray getitem/setitem to work,
        # can be refactored.
        return self

    def eval(self, i):
        return self.parent.eval(self.calc_index(i))

    def getitem(self, item):
        return self.parent.getitem(self.calc_index(item))

    @unwrap_spec(item=int, value=float)
    def descr_setitem(self, space, item, value):
        return self.parent.descr_setitem(space, self.calc_index(item), value)

    def descr_len(self, space):
        return space.wrap(self.find_size())

    def calc_index(self, item):
        raise NotImplementedError

class SingleDimSlice(ViewArray):
    _immutable_fields_ = ["start", "stop", "step", "size"]
    static_signature = Signature()

    def __init__(self, start, stop, step, slice_length, parent, signature):
        ViewArray.__init__(self, parent, signature)
        self.start = start
        self.stop = stop
        self.step = step
        self.size = slice_length

    def find_size(self):
        return self.size

    def calc_index(self, item):
        return (self.start + item * self.step)


class SingleDimArray(BaseArray):
    signature = Signature()

    def __init__(self, size):
        BaseArray.__init__(self)
        self.size = size
        self.storage = lltype.malloc(TP, size, zero=True,
                                     flavor='raw', track_allocation=False)
        # XXX find out why test_zjit explodes with trackign of allocations

    def get_concrete(self):
        return self

    def find_size(self):
        return self.size

    def eval(self, i):
        return self.storage[i]

    def getindex(self, space, item):
        if item >= self.size:
            raise operationerrfmt(space.w_IndexError,
              '%d above array size', item)
        if item < 0:
            item += self.size
        if item < 0:
            raise operationerrfmt(space.w_IndexError,
              '%d below zero', item)
        return item

    def descr_len(self, space):
        return space.wrap(self.size)

    def getitem(self, item):
        return self.storage[item]

    @unwrap_spec(item=int, value=float)
    def descr_setitem(self, space, item, value):
        item = self.getindex(space, item)
        self.invalidated()
        self.storage[item] = value

    def __del__(self):
        lltype.free(self.storage, flavor='raw')

def new_numarray(space, w_size_or_iterable):
    l = space.listview(w_size_or_iterable)
    arr = SingleDimArray(len(l))
    i = 0
    for w_elem in l:
        arr.storage[i] = space.float_w(space.float(w_elem))
        i += 1
    return arr

def descr_new_numarray(space, w_type, w_size_or_iterable):
    return space.wrap(new_numarray(space, w_size_or_iterable))

@unwrap_spec(size=int)
def zeros(space, size):
    return space.wrap(SingleDimArray(size))

@unwrap_spec(size=int)
def ones(space, size):
    arr = SingleDimArray(size)
    for i in xrange(size):
        arr.storage[i] = 1.0
    return space.wrap(arr)

BaseArray.typedef = TypeDef(
    'numarray',
    __new__ = interp2app(descr_new_numarray),

    shape = GetSetProperty(BaseArray.descr_get_shape),

    __len__ = interp2app(BaseArray.descr_len),
    __getitem__ = interp2app(BaseArray.descr_getitem),
    __setitem__ = interp2app(BaseArray.descr_setitem),

    __pos__ = interp2app(BaseArray.descr_pos),
    __neg__ = interp2app(BaseArray.descr_neg),
    __abs__ = interp2app(BaseArray.descr_abs),
    __add__ = interp2app(BaseArray.descr_add),
    __sub__ = interp2app(BaseArray.descr_sub),
    __mul__ = interp2app(BaseArray.descr_mul),
    __div__ = interp2app(BaseArray.descr_div),
    __pow__ = interp2app(BaseArray.descr_pow),
    __mod__ = interp2app(BaseArray.descr_mod),
    __radd__ = interp2app(BaseArray.descr_radd),
    __rsub__ = interp2app(BaseArray.descr_rsub),
    __rmul__ = interp2app(BaseArray.descr_rmul),
    __rdiv__ = interp2app(BaseArray.descr_rdiv),
    __rpow__ = interp2app(BaseArray.descr_rpow),
    __rmod__ = interp2app(BaseArray.descr_rmod),

    mean = interp2app(BaseArray.descr_mean),
    sum = interp2app(BaseArray.descr_sum),
    prod = interp2app(BaseArray.descr_prod),
    max = interp2app(BaseArray.descr_max),
    min = interp2app(BaseArray.descr_min),
    argmax = interp2app(BaseArray.descr_argmax),
    argmin = interp2app(BaseArray.descr_argmin),
    all = interp2app(BaseArray.descr_all),
    any = interp2app(BaseArray.descr_any),
    dot = interp2app(BaseArray.descr_dot),
)
