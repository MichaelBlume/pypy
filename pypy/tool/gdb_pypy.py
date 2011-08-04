"""
Some convenience macros for gdb.  If you have pypy in your path, you can simply do:

(gdb) python import pypy.tool.gdb_pypy

Or, alternatively:

(gdb) python execfile('/path/to/gdb_pypy.py')
"""

import sys
import os.path

try:
    # when running inside gdb
    from gdb import Command
except ImportError:
    # whenn running outside gdb: mock class for testing
    class Command(object):
        def __init__(self, name, command_class):
            pass


def find_field_with_suffix(val, suffix):
    """
    Return ``val[field]``, where ``field`` is the only one whose name ends
    with ``suffix``.  If there is no such field, or more than one, raise KeyError.
    """
    names = []
    for field in val.type.fields():
        if field.name.endswith(suffix):
            names.append(field.name)
    #
    if len(names) == 1:
        return val[names[0]]
    elif len(names) == 0:
        raise KeyError, "cannot find field *%s" % suffix
    else:
        raise KeyError, "too many matching fields: %s" % ', '.join(names)

def lookup(val, suffix):
    """
    Lookup a field which ends with ``suffix`` following the rpython struct
    inheritance hierarchy (i.e., looking both at ``val`` and
    ``val['*_super']``, recursively.
    """
    try:
        return find_field_with_suffix(val, suffix)
    except KeyError:
        baseobj = find_field_with_suffix(val, '_super')
        return lookup(baseobj, suffix)


class RPyType(Command):
    """
    Prints the RPython type of the expression (remember to dereference it!)
    It assumes to find ``typeids.txt`` in the current directory.
    E.g.:

    (gdb) rpy_type *l_v123
    GcStruct pypy.foo.Bar { super, inst_xxx, inst_yyy }
    """

    prog2typeids = {}
 
    def __init__(self, gdb=None):
        # dependency injection, for tests
        if gdb is None:
            import gdb
        self.gdb = gdb
        Command.__init__(self, "rpy_type", self.gdb.COMMAND_NONE)

    def invoke(self, arg, from_tty):
        # some magic code to automatically reload the python file while developing
        ## from pypy.tool import gdb_pypy
        ## reload(gdb_pypy)
        ## gdb_pypy.RPyType.prog2typeids = self.prog2typeids # persist the cache
        ## self.__class__ = gdb_pypy.RPyType
        print self.do_invoke(arg, from_tty)

    def do_invoke(self, arg, from_tty):
        obj = self.gdb.parse_and_eval(arg)
        hdr = lookup(obj, '_gcheader')
        tid = hdr['h_tid']
        offset = tid & 0xFFFFFFFF # 64bit only
        offset = int(offset) # convert from gdb.Value to python int
        typeids = self.get_typeids()
        if offset in typeids:
            return typeids[offset]
        else:
            return 'Cannot find the type with offset %d' % offset

    def get_typeids(self):
        progspace = self.gdb.current_progspace()
        try:
            return self.prog2typeids[progspace]
        except KeyError:
            typeids = self.load_typeids(progspace)
            self.prog2typeids[progspace] = typeids
            return typeids

    def load_typeids(self, progspace):
        """
        Returns a mapping offset --> description
        """
        exename = progspace.filename
        root = os.path.dirname(exename)
        typeids_txt = os.path.join(root, 'typeids.txt')
        print 'loading', typeids_txt
        typeids = {}
        for line in open(typeids_txt):
            member, descr = map(str.strip, line.split(None, 1))
            expr = "((char*)(&pypy_g_typeinfo.%s)) - (char*)&pypy_g_typeinfo" % member
            offset = int(self.gdb.parse_and_eval(expr))
            typeids[offset] = descr
        return typeids

try:
    import gdb
    RPyType() # side effects
except ImportError:
    pass
