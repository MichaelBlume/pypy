
/**************************************************************/
/***  this is included before any code produced by genc.py  ***/


#ifdef PYPY_STANDALONE
#  include "src/commondefs.h"
#endif

#ifdef _WIN32
#  include <io.h>   /* needed, otherwise _lseeki64 truncates to 32-bits (??) */
#endif

#include "thread.h"   /* needs to be included early to define the
                         struct RPyOpaque_ThreadLock */

#include <stddef.h>


#ifdef __GNUC__       /* other platforms too, probably */
typedef _Bool bool_t;
#else
typedef unsigned char bool_t;
#endif


#include "src/align.h"
