from pypy.interpreter.mixedmodule import MixedModule


class Module(MixedModule):
    applevel_name = 'numpypy'

    interpleveldefs = {
        'array': 'interp_numarray.NDimArray',
        'dtype': 'interp_dtype.W_Dtype',
        'ufunc': 'interp_ufuncs.W_Ufunc',

        'zeros': 'interp_numarray.zeros',
        'empty': 'interp_numarray.zeros',
        'ones': 'interp_numarray.ones',
        'dot': 'interp_numarray.dot',
        'fromstring': 'interp_support.fromstring',
        'flatiter': 'interp_numarray.W_FlatIterator',

        'True_': 'space.w_True',
        'False_': 'space.w_False',
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
