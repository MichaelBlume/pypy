from pypy.interpreter.error import OperationError
from pypy.objspace.std.model import registerimplementation, W_Object
from pypy.objspace.std.register_all import register_all
from pypy.objspace.std.inttype import wrapint
from pypy.objspace.std.intobject import W_IntObject
from pypy.objspace.std.floatobject import W_FloatObject
from pypy.objspace.std.stringobject import W_StringObject
from pypy.objspace.std.sliceobject import W_SliceObject, normalize_simple_slice
from pypy.objspace.std.tupleobject import W_TupleObject
from pypy.objspace.std import slicetype
from pypy.rlib.rarithmetic import intmask
from pypy.rlib.objectmodel import compute_hash


class NotSpecialised(Exception):
    pass         
            
def makespecialisedtuple(space, list_w):            
    if len(list_w) == 2:
        w_item0 = list_w[0]
        w_item1 = list_w[1]
        if space.type(w_item0) == space.w_int and space.type(w_item1) == space.w_int:
                val0 = space.int_w(w_item0)
                val1 = space.int_w(w_item1)
                return W_SpecialisedTupleObjectIntInt(space, val0, val1)
    raise NotSpecialised                        

class W_SpecialisedTupleObject(W_Object):
    from pypy.objspace.std.tupletype import tuple_typedef as typedef
    __slots__ = []

    def tolist(self):
        raise NotImplementedError

    def _tolistunwrapped(self):
        raise NotImplementedError

    def length(self):
        raise NotImplementedError

    def getitem(self, index):
        raise NotImplementedError

    def hash(self, space):
        raise NotImplementedError

    def eq(self, space, w_other):
        raise NotImplementedError

    def setitem(self, index, w_item):
        raise NotImplementedError

    def unwrap(w_tuple, space):
        return tuple(self.tolist)
                        
def make_specialised_class(class_name, type0, type1):
    class cls(W_SpecialisedTupleObject):
        def __init__(self, space, val0, val1):
            assert isinstance(val0, type0)
            assert isinstance(val1, type1)
            self.space = space
            self.val0 = val0
            self.val1 = val1
    
        def length(self):
            return 2
    
        def tolist(self):
            return [self.space.wrap(self.val0), self.space.wrap(self.val1)]
            
        def hash(self, space):
            mult = 1000003
            x = 0x345678
            z = 2
            for val in [self.val0, self.val1]:
                y = compute_hash(val)		
                x = (x ^ y) * mult
                z -= 1
                mult += 82520 + z + z
            x += 97531
            return space.wrap(intmask(x))
    
        def eq(self, space, w_other):
            if w_other.length() != 2:
                return space.w_False
            if self.val0 == w_other.val0 and self.val1 == w_other.val1:	#xxx
                return space.w_True
            else:
                return space.w_False
    
        def getitem(self, index):
            if index == 0:
                return self.space.wrap(self.val0)
            if index == 1:
                return self.space.wrap(self.val1)
            raise IndexError
    cls.__name__ = class_name      
    return cls
    
    
W_SpecialisedTupleObjectIntInt = make_specialised_class('W_SpecialisedTupleObjectIntInt', int,int)
    
registerimplementation(W_SpecialisedTupleObject)

def delegate_SpecialisedTuple2Tuple(space, w_specialised):
    return W_TupleObject(w_specialised.tolist())

def len__SpecialisedTuple(space, w_tuple):
    return space.wrap(w_tuple.length())

def getitem__SpecialisedTuple_ANY(space, w_tuple, w_index):
    index = space.getindex_w(w_index, space.w_IndexError, "tuple index")
    if index < 0:
        index += w_tuple.length()
    try:
        return w_tuple.getitem(index)
    except IndexError:
        raise OperationError(space.w_IndexError,
                             space.wrap("tuple index out of range"))

def eq__SpecialisedTuple_SpecialisedTuple(space, w_tuple1, w_tuple2):
    return w_tuple1.eq(space, w_tuple2)

def hash__SpecialisedTuple(space, w_tuple):
    return w_tuple.hash(space)

from pypy.objspace.std import tupletype
register_all(vars(), tupletype)
