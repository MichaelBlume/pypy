import pypy.module.cppyy.capi as capi

from pypy.interpreter.error import OperationError
from pypy.interpreter.gateway import ObjSpace, interp2app
from pypy.interpreter.typedef import TypeDef, interp_attrproperty
from pypy.interpreter.baseobjspace import Wrappable

from pypy.rpython.lltypesystem import rffi, lltype

from pypy.rlib import libffi
from pypy.rlib import jit, debug

from pypy.module.cppyy import converter, executor, helper

class FastCallNotPossible(Exception):
    pass

NULL_VOIDP  = lltype.nullptr(rffi.VOIDP.TO)

def load_lib(space, name):
    # TODO: allow open in RTLD_GLOBAL mode
    cdll = libffi.CDLL(name) #, 0x100 | 0x02)
    return W_CPPLibrary(space, cdll)
load_lib.unwrap_spec = [ObjSpace, str]

class State(object):
    def __init__(self, space):
        self.cpptype_cache = { "void" : W_CPPType(space, "void", NULL_VOIDP) }
        self.cpptemplatetype_cache = {}

def type_byname(space, name):
    state = space.fromcache(State)
    try:
        return state.cpptype_cache[name]
    except KeyError:
        pass

    handle = capi.c_get_typehandle(name)
    if handle:
        final_name = capi.charp2str_free(capi.c_final_name(handle))
        if capi.c_is_namespace(handle):
            cpptype = W_CPPNamespace(space, final_name, handle)
        else:
            cpptype = W_CPPType(space, final_name, handle)
        state.cpptype_cache[name] = cpptype
        cpptype._find_methods()
        cpptype._find_data_members()
        return cpptype

    return None
type_byname.unwrap_spec = [ObjSpace, str]

def template_byname(space, name):
    state = space.fromcache(State)
    try:
        return state.cpptemplatetype_cache[name]
    except KeyError:
        pass

    handle = capi.c_get_templatehandle(name)
    if handle:
        template = W_CPPTemplateType(space, name, handle)
        state.cpptype_cache[name] = template
        return template

    return None
template_byname.unwrap_spec = [ObjSpace, str]


class W_CPPLibrary(Wrappable):
    _immutable_ = True

    def __init__(self, space, cdll):
        self.cdll = cdll
        self.space = space

W_CPPLibrary.typedef = TypeDef(
    'CPPLibrary',
)

@jit.elidable_promote()
def get_methptr_getter(handle, method_index):
    return capi.c_get_methptr_getter(handle, method_index)


class CPPMethod(object):
    """ A concrete function after overloading has been resolved """
    _immutable_ = True
    _immutable_fields_ = ["arg_types[*]", "arg_converters[*]"]
    
    def __init__(self, cpptype, method_index, result_type, arg_types, args_required):
        self.cpptype = cpptype
        self.space = cpptype.space
        self.method_index = method_index
        self.arg_types = arg_types
        self.args_required = args_required
        self.executor = executor.get_executor(self.space, result_type)
        self.arg_converters = None
        methgetter = get_methptr_getter(self.cpptype.handle,
                                        self.method_index)
        self.methgetter = methgetter
        self._libffifunc_cache = {}

    def call(self, cppthis, args_w):
        assert lltype.typeOf(cppthis) == rffi.VOIDP
        if self.executor is None:
            raise OperationError(self.space.w_TypeError,
                                 self.space.wrap("return type not handled"))

        if self.methgetter and cppthis: # only for methods
            try:
                return self.do_fast_call(cppthis, args_w)
            except FastCallNotPossible:
                pass

        args = self.prepare_arguments(args_w)
        try:
            return self.executor.execute(self.space, self, cppthis, len(args_w), args)
        finally:
            self.free_arguments(args, len(args_w))

    @jit.unroll_safe
    def do_fast_call(self, cppthis, args_w):
        space = self.space
        # XXX factor out
        if len(self.arg_types) < len(args_w) or len(args_w) < self.args_required:
            raise OperationError(space.w_TypeError, space.wrap("wrong number of args"))
        if self.arg_converters is None:
            self._build_converters()
        jit.promote(self)
        funcptr = self.methgetter(rffi.cast(rffi.VOIDP, cppthis))
        libffi_func = self._get_libffi_func(funcptr)
        if not libffi_func:
            raise FastCallNotPossible

        argchain = libffi.ArgChain()
        argchain.arg(cppthis)
        for i in range(len(args_w)):
            conv = self.arg_converters[i]
            w_arg = args_w[i]
            conv.convert_argument_libffi(space, w_arg, argchain)
        return self.executor.execute_libffi(space, libffi_func, argchain)

    @jit.elidable_promote()
    def _get_libffi_func(self, funcptr):
        key = rffi.cast(rffi.LONG, funcptr)
        if key in self._libffifunc_cache:
            return self._libffifunc_cache[key]
        argtypes_libffi = [conv.libffitype for conv in self.arg_converters
                              if conv.libffitype]
        if (len(argtypes_libffi) == len(self.arg_converters) and
                self.executor.libffitype):
            # add c++ this to the arguments
            libffifunc = libffi.Func("XXX",
                                     [libffi.types.pointer] + argtypes_libffi,
                                     self.executor.libffitype, funcptr)
        else:
            libffifunc = None
        self._libffifunc_cache[key] = libffifunc
        return libffifunc

    def _build_converters(self):
        self.arg_converters = [converter.get_converter(self.space, arg_type)
                                   for arg_type in self.arg_types]

    @jit.unroll_safe
    def prepare_arguments(self, args_w):
        jit.promote(self)
        space = self.space
        if len(self.arg_types) < len(args_w) or len(args_w) < self.args_required:
            raise OperationError(space.w_TypeError, space.wrap("wrong number of args"))
        if self.arg_converters is None:
            self._build_converters()
        args = capi.c_allocate_function_args(len(args_w))
        stride = capi.c_function_arg_sizeof()
        for i in range(len(args_w)):
            conv = self.arg_converters[i]
            w_arg = args_w[i]
            try:
                arg_i = lltype.direct_ptradd(rffi.cast(rffi.CCHARP, args), i*stride)
                conv.convert_argument(space, w_arg, rffi.cast(rffi.VOIDP, arg_i))
            except:
                # fun :-(
                for j in range(i):
                    conv = self.arg_converters[j]
                    arg_j = lltype.direct_ptradd(rffi.cast(rffi.CCHARP, args), j*stride)
                    conv.free_argument(rffi.cast(rffi.VOIDP, arg_j))
                capi.c_deallocate_function_args(args)
                raise
        return args

    @jit.unroll_safe
    def free_arguments(self, args, nargs):
        stride = capi.c_function_arg_sizeof()
        for i in range(nargs):
            conv = self.arg_converters[i]
            arg_i = lltype.direct_ptradd(rffi.cast(rffi.CCHARP, args), i*stride)
            conv.free_argument(rffi.cast(rffi.VOIDP, arg_i))
        capi.c_deallocate_function_args(args)

    def __repr__(self):
        return "CPPFunction(%s, %s, %r, %s)" % (
            self.cpptype, self.method_index, self.executor, self.arg_types)

    def _freeze_(self):
        assert 0, "you should never have a pre-built instance of this!"


class CPPFunction(CPPMethod):
    _immutable_ = True

    def call(self, cppthis, args_w):
        assert lltype.typeOf(cppthis) == rffi.VOIDP
        if self.executor is None:
            raise OperationError(self.space.w_TypeError,
                                 self.space.wrap("return type not handled"))

        assert not cppthis
        args = self.prepare_arguments(args_w)
        try:
            return self.executor.execute(self.space, self, NULL_VOIDP,
                                         len(args_w), args)
        finally:
            self.free_arguments(args, len(args_w))
 

class CPPConstructor(CPPMethod):
    _immutable_=True

    def call(self, cppthis, args_w):
        assert not cppthis
        newthis = capi.c_allocate(self.cpptype.handle)
        assert lltype.typeOf(newthis) == rffi.VOIDP
        try:
            CPPMethod.call(self, newthis, args_w)
        except Exception, e:
            capi.c_deallocate(self.cpptype.handle, newthis)
            raise
        return W_CPPInstance(self.space, self.cpptype, newthis, True)


class W_CPPOverload(Wrappable):
    _immutable_ = True
    _immutable_fields_ = ["func_name", "functions[*]"]

    def __init__(self, space, func_name, functions):
        self.space = space
        self.func_name = func_name
        self.functions = debug.make_sure_not_resized(functions)

    def is_static(self):
        return self.space.wrap(isinstance(self.functions[0], CPPFunction))

    def get_returntype(self):
        return self.space.wrap(self.functions[0].executor.name)

    @jit.unroll_safe
    def call(self, cppinstance, args_w):
        if cppinstance:
            cppthis = cppinstance.rawobject
        else:
            cppthis = NULL_VOIDP
        assert lltype.typeOf(cppthis) == rffi.VOIDP

        space = self.space
        errmsg = 'None of the overloads matched:'
        jit.promote(self)
        for i in range(len(self.functions)):
            cppyyfunc = self.functions[i]
            try:
                cppresult = cppyyfunc.call(cppthis, args_w)
                if cppinstance and isinstance(cppresult, W_CPPInstance):
                    if cppresult.rawobject == cppinstance.rawobject:
                        return cppinstance  # recycle object to preserve identity
                return cppresult
            except OperationError, e:
                if not e.match(space, space.w_TypeError):
                    raise
                errmsg += '\n\t'+str(e)
            except KeyError:
                pass

        raise OperationError(space.w_TypeError, space.wrap(errmsg))

    def __repr__(self):
        return "W_CPPOverload(%s, %s)" % (self.func_name, self.functions)

W_CPPOverload.typedef = TypeDef(
    'CPPOverload',
    is_static = interp2app(W_CPPOverload.is_static, unwrap_spec=['self']),
    get_returntype = interp2app(W_CPPOverload.get_returntype, unwrap_spec=['self']),
)


class W_CPPDataMember(Wrappable):
    _immutable_=True
    _immutable_fields_ = ["converter", "offset"]

    def __init__(self, space, type_name, offset):
        self.space = space
        self.converter = converter.get_converter(self.space, type_name)
        self.offset = offset

    def is_static(self):
        return self.space.w_False

    def __get__(self, args_w):
        return self.converter.from_memory(self.space, args_w[0], self.offset)

    def __set__(self, args_w):
        self.converter.to_memory(self.space, args_w[0], args_w[1], self.offset)
        return self.space.w_None

W_CPPDataMember.typedef = TypeDef(
    'CPPDataMember',
    is_static = interp2app(W_CPPDataMember.is_static, unwrap_spec=['self']),
    __get__ = interp2app(W_CPPDataMember.__get__, unwrap_spec=['self', 'args_w']),
    __set__ = interp2app(W_CPPDataMember.__set__, unwrap_spec=['self', 'args_w']),
)


class W_CPPStaticDataMember(W_CPPDataMember):
    _immutable_=True

    def is_static(self):
        return self.space.w_True

    def __get__(self, args_w):
        return self.converter.from_memory(self.space, self.space.w_None, self.offset)

    def __set__(self, args_w):
        self.converter.to_memory(self.space, self.space.w_None, args_w[1], self.offset)
        return self.space.w_None

W_CPPStaticDataMember.typedef = TypeDef(
    'CPPStaticDataMember',
    is_static = interp2app(W_CPPStaticDataMember.is_static, unwrap_spec=['self']),
    __get__ = interp2app(W_CPPStaticDataMember.__get__, unwrap_spec=['self', 'args_w']),
    __set__ = interp2app(W_CPPStaticDataMember.__set__, unwrap_spec=['self', 'args_w']),
)


class W_CPPScope(Wrappable):
    _immutable_fields_ = ["name", "handle"]

    def __init__(self, space, name, handle):
        self.space = space
        self.name = name
        self.handle = handle
        self.methods = {}
        # Do not call "self._find_methods()" here, so that a distinction can
        #  be made between testing for existence (i.e. existence in the cache
        #  of classes) and actual use. Point being that a class can use itself,
        #  e.g. as a return type or an argument to one of its methods.

        self.data_members = {}
        # Idem self.methods: a type could hold itself by pointer.

    def _find_methods(self):
        num_methods = capi.c_num_methods(self.handle)
        args_temp = {}
        for i in range(num_methods):
            method_name = capi.charp2str_free(capi.c_method_name(self.handle, i))
            pymethod_name = helper.map_operator_name(
                    method_name, capi.c_method_num_args(self.handle, i),
                    capi.charp2str_free(capi.c_method_result_type(self.handle, i)))
            if not pymethod_name in self.methods:
                cppfunction = self._make_cppfunction(i)
                overload = args_temp.setdefault(pymethod_name, [])
                overload.append(cppfunction)
        for name, functions in args_temp.iteritems():
            overload = W_CPPOverload(self.space, name, functions[:])
            self.methods[name] = overload

    def get_method_names(self):
        return self.space.newlist([self.space.wrap(name) for name in self.methods])

    @jit.elidable_promote('0')
    def get_overload(self, name):
        try:
            return self.methods[name]
        except KeyError:
            raise OperationError(
                self.space.w_AttributeError,
                self.space.wrap(str("class %s has no attribute %s" % (self.name, name))))

    def get_data_member_names(self):
        return self.space.newlist([self.space.wrap(name) for name in self.data_members])

    @jit.elidable_promote('0')
    def get_data_member(self, name):
        try:
            return self.data_members[name]
        except KeyError:
            raise OperationError(
                self.space.w_AttributeError,
                self.space.wrap(str("class %s has no attribute %s" % (self.name, name))))

    def invoke(self, overload, args_w):
        return overload.call(None, args_w)

W_CPPScope.typedef = TypeDef(
    'CPPScope',
    get_method_names = interp2app(W_CPPScope.get_method_names, unwrap_spec=['self']),
    get_overload = interp2app(W_CPPScope.get_overload, unwrap_spec=['self', str]),
    get_data_member_names = interp2app(W_CPPScope.get_data_member_names, unwrap_spec=['self']),
    get_data_member = interp2app(W_CPPScope.get_data_member, unwrap_spec=['self', str]),
    invoke = interp2app(W_CPPScope.invoke, unwrap_spec=['self', W_CPPOverload, 'args_w']),
)


# For now, keep namespaces and classes separate as namespaces are extensible
# with info from multiple dictionaries and do not need to bother with meta
# classes for inheritance. Both are python classes, though, and refactoring
# may be in order at some point.
class W_CPPNamespace(W_CPPScope):

    def _make_cppfunction(self, method_index):
        result_type = capi.charp2str_free(capi.c_method_result_type(self.handle, method_index))
        num_args = capi.c_method_num_args(self.handle, method_index)
        args_required = capi.c_method_req_args(self.handle, method_index)
        argtypes = []
        for i in range(num_args):
            argtype = capi.charp2str_free(capi.c_method_arg_type(self.handle, method_index, i))
            argtypes.append(argtype)
        return CPPFunction(self, method_index, result_type, argtypes, args_required)

    def _find_data_members(self):
        num_data_members = capi.c_num_data_members(self.handle)
        for i in range(num_data_members):
            data_member_name = capi.charp2str_free(capi.c_data_member_name(self.handle, i))
            if not data_member_name in self.data_members:
                type_name = capi.charp2str_free(capi.c_data_member_type(self.handle, i))
                offset = capi.c_data_member_offset(self.handle, i)
                data_member = W_CPPStaticDataMember(self.space, type_name, offset)
                self.data_members[data_member_name] = data_member

    def update(self):
        self._find_methods()
        self._find_data_members()

    def is_namespace(self):
        return self.space.w_True

W_CPPNamespace.typedef = TypeDef(
    'CPPNamespace',
    update = interp2app(W_CPPNamespace.update, unwrap_spec=['self']),
    get_method_names = interp2app(W_CPPNamespace.get_method_names, unwrap_spec=['self']),
    get_overload = interp2app(W_CPPNamespace.get_overload, unwrap_spec=['self', str]),
    get_data_member_names = interp2app(W_CPPNamespace.get_data_member_names, unwrap_spec=['self']),
    get_data_member = interp2app(W_CPPNamespace.get_data_member, unwrap_spec=['self', str]),
    is_namespace = interp2app(W_CPPNamespace.is_namespace, unwrap_spec=['self']),
    invoke = interp2app(W_CPPNamespace.invoke, unwrap_spec=['self', W_CPPOverload, 'args_w']),
)


class W_CPPType(W_CPPScope):

    def _make_cppfunction(self, method_index):
        result_type = capi.charp2str_free(capi.c_method_result_type(self.handle, method_index))
        num_args = capi.c_method_num_args(self.handle, method_index)
        args_required = capi.c_method_req_args(self.handle, method_index)
        argtypes = []
        for i in range(num_args):
            argtype = capi.charp2str_free(capi.c_method_arg_type(self.handle, method_index, i))
            argtypes.append(argtype)
        if capi.c_is_constructor(self.handle, method_index):
            cls = CPPConstructor
        elif capi.c_is_staticmethod(self.handle, method_index):
            cls = CPPFunction
        else:
            cls = CPPMethod
        return cls(self, method_index, result_type, argtypes, args_required)

    def _find_data_members(self):
        num_data_members = capi.c_num_data_members(self.handle)
        for i in range(num_data_members):
            data_member_name = capi.charp2str_free(capi.c_data_member_name(self.handle, i))
            type_name = capi.charp2str_free(capi.c_data_member_type(self.handle, i))
            offset = capi.c_data_member_offset(self.handle, i)
            if capi.c_is_staticdata(self.handle, i):
                data_member = W_CPPStaticDataMember(self.space, type_name, offset)
            else:
                data_member = W_CPPDataMember(self.space, type_name, offset)
            self.data_members[data_member_name] = data_member

    def is_namespace(self):
        return self.space.w_False

    def get_base_names(self):
        bases = []
        num_bases = capi.c_num_bases(self.handle)
        for i in range(num_bases):
            base_name = capi.charp2str_free(capi.c_base_name(self.handle, i))
            bases.append(self.space.wrap(base_name))
        return self.space.newlist(bases)

    def construct(self, args_w):
        try:
            overload = self.get_overload(self.name)
        except OperationError, e:
            if e.match(self.space, self.space.w_AttributeError):
                raise OperationError(self.space.w_TypeError,
                                     self.space.wrap("%s is abstract" % self.name))
            raise

        return overload.call(None, args_w)

W_CPPType.typedef = TypeDef(
    'CPPType',
    type_name = interp_attrproperty('name', W_CPPType),
    get_base_names = interp2app(W_CPPType.get_base_names, unwrap_spec=['self']),
    get_method_names = interp2app(W_CPPType.get_method_names, unwrap_spec=['self']),
    get_overload = interp2app(W_CPPType.get_overload, unwrap_spec=['self', str]),
    get_data_member_names = interp2app(W_CPPType.get_data_member_names, unwrap_spec=['self']),
    get_data_member = interp2app(W_CPPType.get_data_member, unwrap_spec=['self', str]),
    is_namespace = interp2app(W_CPPType.is_namespace, unwrap_spec=['self']),
    invoke = interp2app(W_CPPType.invoke, unwrap_spec=['self', W_CPPOverload, 'args_w']),
    construct = interp2app(W_CPPType.construct, unwrap_spec=['self', 'args_w']),
)


class W_CPPTemplateType(Wrappable):
    _immutable_fields_ = ["name", "handle"]

    def __init__(self, space, name, handle):
        self.space = space
        self.name = name
        self.handle = handle

    def __call__(self, args_w):
        # TODO: this is broken but unused (see pythonify.py)
        fullname = "".join([self.name, '<', self.space.str_w(args_w[0]), '>'])
        return type_byname(self.space, fullname)

W_CPPTemplateType.typedef = TypeDef(
    'CPPTemplateType',
    __call__ = interp2app(W_CPPTemplateType.__call__, unwrap_spec=['self', 'args_w']),
)


class W_CPPInstance(Wrappable):
    _immutable_fields_ = ["cppclass"]

    def __init__(self, space, cppclass, rawobject, python_owns):
        self.space = space
        self.cppclass = cppclass
        assert lltype.typeOf(rawobject) == rffi.VOIDP
        self.rawobject = rawobject
        self.python_owns = python_owns

    def _nullcheck(self):
        if not self.rawobject:
            raise OperationError(self.space.w_ReferenceError,
                                 self.space.wrap("trying to access a NULL pointer"))

    def invoke(self, overload, args_w):
        self._nullcheck()
        return overload.call(self, args_w)

    def destruct(self):
        if self.rawobject:
            capi.c_destruct(self.cppclass.handle, self.rawobject)
            self.rawobject = NULL_VOIDP

    def __del__(self):
        if self.python_owns:
            self.destruct()

W_CPPInstance.typedef = TypeDef(
    'CPPInstance',
    cppclass = interp_attrproperty('cppclass', W_CPPInstance),
    invoke = interp2app(W_CPPInstance.invoke, unwrap_spec=['self', W_CPPOverload, 'args_w']),
    destruct = interp2app(W_CPPInstance.destruct, unwrap_spec=['self']),
)
