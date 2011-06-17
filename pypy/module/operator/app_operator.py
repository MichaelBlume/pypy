'''NOT_RPYTHON: because of attrgetter and itemgetter
Operator interface.

This module exports a set of operators as functions. E.g. operator.add(x,y) is
equivalent to x+y.
'''
from __pypy__ import builtinify

def countOf(a,b): 
    'countOf(a, b) -- Return the number of times b occurs in a.'
    count = 0
    for x in a:
        if x == b:
            count += 1
    return count

def delslice(obj, start, end):
    'delslice(a, b, c) -- Same as del a[b:c].'
    if not isinstance(start, int) or not isinstance(end, int):
        raise TypeError("an integer is expected")
    del obj[start:end]
__delslice__ = delslice

def getslice(a, start, end):
    'getslice(a, b, c) -- Same as a[b:c].'
    if not isinstance(start, int) or not isinstance(end, int):
        raise TypeError("an integer is expected")
    return a[start:end] 
__getslice__ = getslice

def indexOf(a, b):
    'indexOf(a, b) -- Return the first index of b in a.'
    index = 0
    for x in a:
        if x == b:
            return index
        index += 1
    raise ValueError, 'sequence.index(x): x not in sequence'

# XXX the following is approximative
def isMappingType(obj,):
    'isMappingType(a) -- Return True if a has a mapping type, False otherwise.'
    # XXX this is fragile and approximative anyway
    return hasattr(obj, '__getitem__') and hasattr(obj, 'keys')

def isNumberType(obj,):
    'isNumberType(a) -- Return True if a has a numeric type, False otherwise.'
    return hasattr(obj, '__int__') or hasattr(obj, '__float__')

def isSequenceType(obj,):
    'isSequenceType(a) -- Return True if a has a sequence type, False otherwise.'
    return hasattr(obj, '__getitem__') and not hasattr(obj, 'keys')

def repeat(obj, num):
    'repeat(a, b) -- Return a * b, where a is a sequence, and b is an integer.'
    if not isinstance(num, (int, long)):
        raise TypeError, 'an integer is required'
    if not isSequenceType(obj):
        raise TypeError, "non-sequence object can't be repeated"

    return obj * num

__repeat__ = repeat

def setslice(a, b, c, d):
    'setslice(a, b, c, d) -- Same as a[b:c] = d.'
    a[b:c] = d 
__setslice__ = setslice


def attrgetter(attr, *attrs):
    if attrs:
        getters = [single_attr_getter(a) for a in (attr,) + attrs]
        def getter(obj):
            return tuple([getter(obj) for getter in getters])
    else:
        getter = single_attr_getter(attr)
    return builtinify(getter)

def single_attr_getter(attr):
    if not isinstance(attr, str):
        if not isinstance(attr, unicode):
            def _raise_typeerror(obj):
                raise TypeError("argument must be a string, not %r" %
                                (type(attr).__name__,))
            return _raise_typeerror
        attr = attr.encode('ascii')
    #
    def make_getter(name, prevfn=None):
        if prevfn is None:
            def getter(obj):
                return getattr(obj, name)
        else:
            def getter(obj):
                return getattr(prevfn(obj), name)
        return getter
    #
    last = 0
    getter = None
    while True:
        dot = attr.find(".", last)
        if dot < 0: break
        getter = make_getter(attr[last:dot], getter)
        last = dot + 1
    return make_getter(attr[last:], getter)


class itemgetter(object):

    def __init__(self, item, *args):
        self.items = args
        self.item = item

    def __call__(self, obj):
        result = obj[self.item]

        if self.items:
            list = [result] + [obj[item] for item in self.items]
            return tuple(list)

        return result

class methodcaller(object):

    def __init__(self, method_name, *args, **kwargs):
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, obj):
        return getattr(obj, self.method_name)(*self.args, **self.kwargs)
