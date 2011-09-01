# Constants that depend on whether we are on 32-bit or 64-bit

import sys
if sys.maxint == (2**31 - 1):
    WORD = 4
    IS_PPC_32 = True
    IS_PPC_64 = False
else:
    WORD = 8
    IS_PPC_32 = False
    IS_PPC_64 = True

NONVOLATILES    = [2] + range(13, 32)
VOLATILES       = [0] + range(3, 13)
