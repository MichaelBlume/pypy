from pypy.interpreter.mixedmodule import MixedModule


class PyPyModule(MixedModule):
    interpleveldefs = {
        'debug_repr': 'interp_extras.debug_repr',
        'remove_invalidates': 'interp_extras.remove_invalidates',
        'set_invalidation': 'interp_extras.set_invalidation',
    }
    appleveldefs = {}

class Module(MixedModule):
    applevel_name = '_numpypy'

    submodules = {
        'pypy': PyPyModule,
    }

    interpleveldefs = {
        'ndarray': 'interp_numarray.W_NDArray',
        'array':   'interp_numarray.descr_array',
    }
    appleveldefs = {}

    # interpleveldefs = {
    #     'ndarray': 'interp_numarray.W_NDArray',
    #     'dtype': 'interp_dtype.W_Dtype',
    #     'ufunc': 'interp_ufuncs.W_Ufunc',

    #     'array': 'interp_numarray.array',
    #     'zeros': 'interp_numarray.zeros',
    #     'empty': 'interp_numarray.zeros',
    #     'ones': 'interp_numarray.ones',
    #     'dot': 'interp_numarray.dot',
    #     'fromstring': 'interp_support.fromstring',
    #     'flatiter': 'interp_numarray.W_FlatIterator',
    #     'isna': 'interp_numarray.isna',
    #     'concatenate': 'interp_numarray.concatenate',
    #     'repeat': 'interp_numarray.repeat',
    #     'where': 'interp_arrayops.where',

    #     'set_string_function': 'appbridge.set_string_function',

    #     'count_reduce_items': 'interp_numarray.count_reduce_items',

    #     'True_': 'types.Bool.True',
    #     'False_': 'types.Bool.False',

    #     'typeinfo': 'interp_dtype.get_dtype_cache(space).w_typeinfo',

    #     'generic': 'interp_boxes.W_GenericBox',
    #     'number': 'interp_boxes.W_NumberBox',
    #     'integer': 'interp_boxes.W_IntegerBox',
    #     'signedinteger': 'interp_boxes.W_SignedIntegerBox',
    #     'unsignedinteger': 'interp_boxes.W_UnsignedIntegerBox',
    #     'bool_': 'interp_boxes.W_BoolBox',
    #     'bool8': 'interp_boxes.W_BoolBox',
    #     'int8': 'interp_boxes.W_Int8Box',
    #     'byte': 'interp_boxes.W_Int8Box',
    #     'uint8': 'interp_boxes.W_UInt8Box',
    #     'ubyte': 'interp_boxes.W_UInt8Box',
    #     'int16': 'interp_boxes.W_Int16Box',
    #     'short': 'interp_boxes.W_Int16Box',
    #     'uint16': 'interp_boxes.W_UInt16Box',
    #     'ushort': 'interp_boxes.W_UInt16Box',
    #     'int32': 'interp_boxes.W_Int32Box',
    #     'intc': 'interp_boxes.W_Int32Box',
    #     'uint32': 'interp_boxes.W_UInt32Box',
    #     'uintc': 'interp_boxes.W_UInt32Box',
    #     'int64': 'interp_boxes.W_Int64Box',
    #     'uint64': 'interp_boxes.W_UInt64Box',
    #     'longlong': 'interp_boxes.W_LongLongBox',
    #     'ulonglong': 'interp_boxes.W_ULongLongBox',
    #     'int_': 'interp_boxes.W_LongBox',
    #     'inexact': 'interp_boxes.W_InexactBox',
    #     'floating': 'interp_boxes.W_FloatingBox',
    #     'float_': 'interp_boxes.W_Float64Box',
    #     'float32': 'interp_boxes.W_Float32Box',
    #     'float64': 'interp_boxes.W_Float64Box',
    #     'intp': 'types.IntP.BoxType',
    #     'uintp': 'types.UIntP.BoxType',
    #     'flexible': 'interp_boxes.W_FlexibleBox',
    #     'character': 'interp_boxes.W_CharacterBox',
    #     'str_': 'interp_boxes.W_StringBox',
    #     'unicode_': 'interp_boxes.W_UnicodeBox',
    #     'void': 'interp_boxes.W_VoidBox',
    # }

    # # ufuncs
    # for exposed, impl in [
    #     ("abs", "absolute"),
    #     ("absolute", "absolute"),
    #     ("add", "add"),
    #     ("arccos", "arccos"),
    #     ("arcsin", "arcsin"),
    #     ("arctan", "arctan"),
    #     ("arctan2", "arctan2"),
    #     ("arccosh", "arccosh"),
    #     ("arcsinh", "arcsinh"),
    #     ("arctanh", "arctanh"),
    #     ("copysign", "copysign"),
    #     ("cos", "cos"),
    #     ("cosh", "cosh"),
    #     ("divide", "divide"),
    #     ("true_divide", "true_divide"),
    #     ("equal", "equal"),
    #     ("exp", "exp"),
    #     ("exp2", "exp2"),
    #     ("expm1", "expm1"),
    #     ("fabs", "fabs"),
    #     ("fmax", "fmax"),
    #     ("fmin", "fmin"),
    #     ("fmod", "fmod"),
    #     ("floor", "floor"),
    #     ("ceil", "ceil"),
    #     ("trunc", "trunc"),
    #     ("greater", "greater"),
    #     ("greater_equal", "greater_equal"),
    #     ("less", "less"),
    #     ("less_equal", "less_equal"),
    #     ("maximum", "maximum"),
    #     ("minimum", "minimum"),
    #     ("multiply", "multiply"),
    #     ("negative", "negative"),
    #     ("not_equal", "not_equal"),
    #     ("radians", "radians"),
    #     ("degrees", "degrees"),
    #     ("deg2rad", "radians"),
    #     ("rad2deg", "degrees"),
    #     ("reciprocal", "reciprocal"),
    #     ("sign", "sign"),
    #     ("signbit", "signbit"),
    #     ("sin", "sin"),
    #     ("sinh", "sinh"),
    #     ("subtract", "subtract"),
    #     ('sqrt', 'sqrt'),
    #     ('square', 'square'),
    #     ("tan", "tan"),
    #     ("tanh", "tanh"),
    #     ('bitwise_and', 'bitwise_and'),
    #     ('bitwise_or', 'bitwise_or'),
    #     ('bitwise_xor', 'bitwise_xor'),
    #     ('bitwise_not', 'invert'),
    #     ('left_shift', 'left_shift'),
    #     ('right_shift', 'right_shift'),
    #     ('invert', 'invert'),
    #     ('isnan', 'isnan'),
    #     ('isinf', 'isinf'),
    #     ('isneginf', 'isneginf'),
    #     ('isposinf', 'isposinf'),
    #     ('isfinite', 'isfinite'),
    #     ('logical_and', 'logical_and'),
    #     ('logical_xor', 'logical_xor'),
    #     ('logical_not', 'logical_not'),
    #     ('logical_or', 'logical_or'),
    #     ('log', 'log'),
    #     ('log2', 'log2'),
    #     ('log10', 'log10'),
    #     ('log1p', 'log1p'),
    #     ('power', 'power'),
    #     ('floor_divide', 'floor_divide'),
    #     ('logaddexp', 'logaddexp'),
    #     ('logaddexp2', 'logaddexp2'),
    # ]:
    #     interpleveldefs[exposed] = "interp_ufuncs.get(space).%s" % impl

    # appleveldefs = {
    #     'average': 'app_numpy.average',
    #     'sum': 'app_numpy.sum',
    #     'min': 'app_numpy.min',
    #     'identity': 'app_numpy.identity',
    #     'max': 'app_numpy.max',
    #     'arange': 'app_numpy.arange',
    # }
