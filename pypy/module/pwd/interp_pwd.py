from pypy.translator.tool.cbuild import ExternalCompilationInfo
from pypy.rpython.tool import rffi_platform
from pypy.rpython.lltypesystem import rffi, lltype
from pypy.interpreter.gateway import interp2app, unwrap_spec
from pypy.interpreter.error import OperationError, operationerrfmt

class CConfig:
    _compilation_info_ = ExternalCompilationInfo(
        includes=['pwd.h']
        )

    uid_t = rffi_platform.SimpleType("uid_t")

    passwd = rffi_platform.Struct(
        'struct passwd',
        [('pw_name', rffi.CCHARP),
         ('pw_passwd', rffi.CCHARP),
         ('pw_uid', rffi.INT),
         ('pw_gid', rffi.INT),
         ('pw_gecos', rffi.CCHARP),
         ('pw_dir', rffi.CCHARP),
         ('pw_shell', rffi.CCHARP),
         ])

config = rffi_platform.configure(CConfig)
passwd_p = lltype.Ptr(config['passwd'])
uid_t = config['uid_t']

c_getpwuid = rffi.llexternal("getpwuid", [uid_t], passwd_p)
c_getpwnam = rffi.llexternal("getpwnam", [rffi.CCHARP], passwd_p)
c_setpwent = rffi.llexternal("setpwent", [], lltype.Void)
c_getpwent = rffi.llexternal("getpwent", [], passwd_p)
c_endpwent = rffi.llexternal("endpwent", [], lltype.Void)

def make_struct_passwd(space, pw):
    w_passwd_struct = space.getattr(space.getbuiltinmodule('pwd'),
                                    space.wrap('struct_passwd'))
    w_tuple = space.newtuple([
        space.wrap(rffi.charp2str(pw.c_pw_name)),
        space.wrap(rffi.charp2str(pw.c_pw_passwd)),
        space.wrap(pw.c_pw_uid),
        space.wrap(pw.c_pw_gid),
        space.wrap(rffi.charp2str(pw.c_pw_gecos)),
        space.wrap(rffi.charp2str(pw.c_pw_dir)),
        space.wrap(rffi.charp2str(pw.c_pw_shell)),
        ])
    return space.call_function(w_passwd_struct, w_tuple)

@unwrap_spec(uid=int)
def getpwuid(space, uid):
    """
    getpwuid(uid) -> (pw_name,pw_passwd,pw_uid,
                      pw_gid,pw_gecos,pw_dir,pw_shell)
    Return the password database entry for the given numeric user ID.
    See pwd.__doc__ for more on password database entries.
    """
    pw = c_getpwuid(uid)
    if not pw:
        raise operationerrfmt(space.w_KeyError,
            "getpwuid(): uid not found: %d", uid)
    return make_struct_passwd(space, pw)

@unwrap_spec(name=str)
def getpwnam(space, name):
    """
    getpwnam(name) -> (pw_name,pw_passwd,pw_uid,
                        pw_gid,pw_gecos,pw_dir,pw_shell)
    Return the password database entry for the given user name.
    See pwd.__doc__ for more on password database entries.
    """
    pw = c_getpwnam(name)
    if not pw:
        raise operationerrfmt(space.w_KeyError,
            "getpwnam(): name not found: %s", name)
    return make_struct_passwd(space, pw)

def getpwall(space):
    users_w = []
    c_setpwent()
    try:
        while True:
            pw = c_getpwent()
            if not pw:
                break
            users_w.append(make_struct_passwd(space, pw))
    finally:
        c_endpwent()
    return space.newlist(users_w)
    
