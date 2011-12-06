from pypy.interpreter.mixedmodule import MixedModule


class Module(MixedModule):
    applevel_name = 'numpypy'

    interpleveldefs = {
        'ndarray': 'interp_numarray.W_NDimArray',
        'dtype': 'interp_dtype.W_Dtype',
        'ufunc': 'interp_ufuncs.W_Ufunc',

        'array': 'interp_numarray.array',
        'zeros': 'interp_numarray.zeros',
        'empty': 'interp_numarray.zeros',
        'ones': 'interp_numarray.ones',
        'dot': 'interp_numarray.dot',
        'fromstring': 'interp_support.fromstring',
        'flatiter': 'interp_numarray.W_FlatIterator',

        'True_': 'types.Bool.True',
        'False_': 'types.Bool.False',

        'generic': 'interp_boxes.W_GenericBox',
        'number': 'interp_boxes.W_NumberBox',
        'integer': 'interp_boxes.W_IntegerBox',
        'signedinteger': 'interp_boxes.W_SignedIntegerBox',
        'bool_': 'interp_boxes.W_BoolBox',
        'int8': 'interp_boxes.W_Int8Box',
        'int16': 'interp_boxes.W_Int16Box',
        'int32': 'interp_boxes.W_Int32Box',
        'int64': 'interp_boxes.W_Int64Box',
        'int_': 'interp_boxes.W_LongBox',
        'inexact': 'interp_boxes.W_InexactBox',
        'floating': 'interp_boxes.W_FloatingBox',
        'float32': 'interp_boxes.W_Float32Box',
        'float64': 'interp_boxes.W_Float64Box',
        'complexfloating': 'interp_boxes.W_ComplexFloatingBox',
        'complex128': 'interp_boxes.W_Complex128Box',
    }

    # ufuncs
    for exposed, impl in [
        ("abs", "absolute"),
        ("absolute", "absolute"),
        ("add", "add"),
        ("arccos", "arccos"),
        ("arcsin", "arcsin"),
        ("arctan", "arctan"),
        ("arcsinh", "arcsinh"),
        ("arctanh", "arctanh"),
        ("copysign", "copysign"),
        ("cos", "cos"),
        ("divide", "divide"),
        ("equal", "equal"),
        ("exp", "exp"),
        ("fabs", "fabs"),
        ("floor", "floor"),
        ("greater", "greater"),
        ("greater_equal", "greater_equal"),
        ("less", "less"),
        ("less_equal", "less_equal"),
        ("maximum", "maximum"),
        ("minimum", "minimum"),
        ("multiply", "multiply"),
        ("negative", "negative"),
        ("not_equal", "not_equal"),
        ("reciprocal", "reciprocal"),
        ("sign", "sign"),
        ("sin", "sin"),
        ("subtract", "subtract"),
        ('sqrt', 'sqrt'),
        ("tan", "tan"),
    ]:
        interpleveldefs[exposed] = "interp_ufuncs.get(space).%s" % impl

    appleveldefs = {
        'average': 'app_numpy.average',
        'mean': 'app_numpy.mean',
        'inf': 'app_numpy.inf',
        'e': 'app_numpy.e',
        'arange': 'app_numpy.arange',
        'reshape': 'app_numpy.reshape',
    }
