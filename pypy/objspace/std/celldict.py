""" A very simple cell dict implementation. The dictionary maps keys to cell.
This ensures that the function (dict, key) -> cell is pure. By itself, this
optimization is not helping at all, but in conjunction with the JIT it can
speed up global lookups a lot."""

from pypy.objspace.std.dictmultiobject import IteratorImplementation
from pypy.objspace.std.dictmultiobject import DictStrategy, _never_equal_to_string
from pypy.objspace.std.dictmultiobject import ObjectDictStrategy
from pypy.rlib import jit, rerased

class ModuleCell(object):
    def __init__(self, w_value=None):
        self.w_value = w_value

    def invalidate(self):
        w_value = self.w_value
        self.w_value = None
        return w_value

    def __repr__(self):
        return "<ModuleCell: %s>" % (self.w_value, )

class ModuleDictStrategy(DictStrategy):

    erase, unerase = rerased.new_erasing_pair("modulecell")
    erase = staticmethod(erase)
    unerase = staticmethod(unerase)

    def __init__(self, space):
        self.space = space

    def get_empty_storage(self):
       return self.erase({})

    def getcell(self, w_dict, key, makenew):
        if makenew or jit.we_are_jitted():
            # when we are jitting, we always go through the pure function
            # below, to ensure that we have no residual dict lookup
            w_dict = jit.promote(w_dict)
            self = jit.promote(self)
            return self._getcell_makenew(w_dict, key)
        return self.unerase(w_dict.dstorage).get(key, None)

    @jit.elidable
    def _getcell_makenew(self, w_dict, key):
        return self.unerase(w_dict.dstorage).setdefault(key, ModuleCell())

    def setitem(self, w_dict, w_key, w_value):
        space = self.space
        if space.is_w(space.type(w_key), space.w_str):
            self.setitem_str(w_dict, self.space.str_w(w_key), w_value)
        else:
            self.switch_to_object_strategy(w_dict)
            w_dict.setitem(w_key, w_value)

    def setitem_str(self, w_dict, key, w_value):
        self.getcell(w_dict, key, True).w_value = w_value

    def setdefault(self, w_dict, w_key, w_default):
        space = self.space
        if space.is_w(space.type(w_key), space.w_str):
            cell = self.getcell(w_dict, space.str_w(w_key), True)
            if cell.w_value is None:
                cell.w_value = w_default
            return cell.w_value
        else:
            self.switch_to_object_strategy(w_dict)
            return w_dict.setdefault(w_key, w_default)

    def delitem(self, w_dict, w_key):
        space = self.space
        w_key_type = space.type(w_key)
        if space.is_w(w_key_type, space.w_str):
            key = space.str_w(w_key)
            cell = self.getcell(w_dict, key, False)
            if cell is None or cell.w_value is None:
                raise KeyError
            # note that we don't remove the cell from self.content, to make
            # sure that a key that was found at any point in the dict, still
            # maps to the same cell later (even if this cell no longer
            # represents a key)
            cell.invalidate()
        elif _never_equal_to_string(space, w_key_type):
            raise KeyError
        else:
            self.switch_to_object_strategy(w_dict)
            w_dict.delitem(w_key)

    def length(self, w_dict):
        # inefficient, but do we care?
        res = 0
        for cell in self.unerase(w_dict.dstorage).itervalues():
            if cell.w_value is not None:
                res += 1
        return res

    def getitem(self, w_dict, w_key):
        space = self.space
        w_lookup_type = space.type(w_key)
        if space.is_w(w_lookup_type, space.w_str):
            return self.getitem_str(w_dict, space.str_w(w_key))

        elif _never_equal_to_string(space, w_lookup_type):
            return None
        else:
            self.switch_to_object_strategy(w_dict)
            return w_dict.getitem(w_key)

    def getitem_str(self, w_dict, key):
        res = self.getcell(w_dict, key, False)
        if res is None:
            return None
        # note that even if the res.w_value is None, the next line is fine
        return res.w_value

    def iter(self, w_dict):
        return ModuleDictIteratorImplementation(self.space, self, w_dict)

    def keys(self, w_dict):
        space = self.space
        iterator = self.unerase(w_dict.dstorage).iteritems
        return [space.wrap(key) for key, cell in iterator()
                    if cell.w_value is not None]

    def values(self, w_dict):
        iterator = self.unerase(w_dict.dstorage).itervalues
        return [cell.w_value for cell in iterator()
                    if cell.w_value is not None]

    def items(self, w_dict):
        space = self.space
        iterator = self.unerase(w_dict.dstorage).iteritems
        return [space.newtuple([space.wrap(key), cell.w_value])
                    for (key, cell) in iterator()
                        if cell.w_value is not None]

    def clear(self, w_dict):
        iterator = self.unerase(w_dict.dstorage).iteritems
        for k, cell in iterator():
            cell.invalidate()

    def popitem(self, w_dict):
        # This is O(n) if called repeatadly, you probably shouldn't be on a
        # Module's dict though
        for k, cell in self.unerase(w_dict.dstorage).iteritems():
            if cell.w_value is not None:
                w_value = cell.w_value
                cell.invalidate()
                return self.space.wrap(k), w_value
        else:
            raise KeyError

    def switch_to_object_strategy(self, w_dict):
        d = self.unerase(w_dict.dstorage)
        strategy = self.space.fromcache(ObjectDictStrategy)
        d_new = strategy.unerase(strategy.get_empty_storage())
        for key, cell in d.iteritems():
            if cell.w_value is not None:
                d_new[self.space.wrap(key)] = cell.w_value
        w_dict.strategy = strategy
        w_dict.dstorage = strategy.erase(d_new)

class ModuleDictIteratorImplementation(IteratorImplementation):
    def __init__(self, space, strategy, dictimplementation):
        IteratorImplementation.__init__(self, space, dictimplementation)
        dict_w = strategy.unerase(dictimplementation.dstorage)
        self.iterator = dict_w.iteritems()

    def next_entry(self):
        for key, cell in self.iterator:
            if cell.w_value is not None:
                return (self.space.wrap(key), cell.w_value)
        else:
            return None, None
