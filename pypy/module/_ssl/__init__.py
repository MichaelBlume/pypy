from pypy.interpreter.mixedmodule import MixedModule

class Module(MixedModule):
    """Implementation module for SSL socket operations.
    See the socket module for documentation."""

    interpleveldefs = {
        'SSLError': 'interp_ssl.get_error(space)',
        '_SSLSocket': 'interp_ssl.SSLSocket',
        '_SSLContext': 'interp_ssl.SSLContext',
        '_test_decode_cert': 'interp_ssl._test_decode_cert',
    }

    appleveldefs = {
    }

    @classmethod
    def buildloaders(cls):
        # init the SSL module
        from pypy.module._ssl.interp_ssl import constants, HAVE_OPENSSL_RAND

        for constant, value in constants.iteritems():
            Module.interpleveldefs[constant] = "space.wrap(%r)" % (value,)

        if HAVE_OPENSSL_RAND:
            Module.interpleveldefs['RAND_add'] = "interp_ssl.RAND_add"
            Module.interpleveldefs['RAND_status'] = "interp_ssl.RAND_status"
            Module.interpleveldefs['RAND_egd'] = "interp_ssl.RAND_egd"

        super(Module, cls).buildloaders()

    def startup(self, space):
        from pypy.rlib.ropenssl import init_ssl
        init_ssl()
        from pypy.module._ssl.interp_ssl import setup_ssl_threads
        setup_ssl_threads()
