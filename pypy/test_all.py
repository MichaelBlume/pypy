#! /usr/bin/env python
"""
PyPy Test runner interface
--------------------------

Running test_all.py is equivalent to running py.test
which you independently install, see
http://pytest.org/getting-started.html

For more information, use test_all.py -h.
"""
import sys, os


if __name__ == '__main__':
    if len(sys.argv) == 1 and os.path.dirname(sys.argv[0]) in '.':
        print >> sys.stderr, __doc__
        sys.exit(2)

    import tool.autopath
    import pytest
    import pytest_cov
    sys.exit(pytest.main(plugins=[pytest_cov]))
