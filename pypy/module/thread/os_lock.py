"""
Python locks, based on true threading locks provided by the OS.
"""

from pypy.module.thread import ll_thread as thread
from pypy.module.thread.error import wrap_thread_error
from pypy.interpreter.baseobjspace import Wrappable
from pypy.interpreter.gateway import interp2app, unwrap_spec
from pypy.interpreter.typedef import TypeDef, make_weakref_descr
from pypy.interpreter.error import OperationError
from pypy.rlib.rarithmetic import r_longlong, ovfcheck

# Force the declaration of the type 'thread.LockType' for RPython
#import pypy.module.thread.rpython.exttable

LONGLONG_MAX = r_longlong(2 ** (r_longlong.BITS-1) - 1)
TIMEOUT_MAX = LONGLONG_MAX


##import sys
##def debug(msg, n):
##    return
##    tb = []
##    try:
##        for i in range(1, 8):
##            tb.append(sys._getframe(i).f_code.co_name)
##    except:
##        pass
##    tb = ' '.join(tb)
##    msg = '| %6d | %d %s | %s\n' % (thread.get_ident(), n, msg, tb)
##    sys.stderr.write(msg)


class Lock(Wrappable):
    "A wrappable box around an interp-level lock object."

    def __init__(self, space):
        self.space = space
        try:
            self.lock = thread.allocate_lock()
        except thread.error:
            raise wrap_thread_error(space, "out of resources")

    @unwrap_spec(blocking=int, timeout=float)
    def descr_lock_acquire(self, space, blocking=1, timeout=-1.0):
        """Lock the lock.  Without argument, this blocks if the lock is already
locked (even by the same thread), waiting for another thread to release
the lock, and return None once the lock is acquired.
With an argument, this will only block if the argument is true,
and the return value reflects whether the lock is acquired.
The blocking operation is interruptible."""
        if not blocking and timeout != -1.0:
            raise OperationError(space.w_ValueError, space.wrap(
                    "can't specify a timeout for a non-blocking call"))
        if timeout < 0.0 and timeout != -1.0:
            raise OperationError(space.w_ValueError, space.wrap(
                    "timeout value must be strictly positive"))
        if not blocking:
            microseconds = 0
        elif timeout == -1.0:
            microseconds = -1
        else:
            timeout *= 1e6
            if timeout > float(TIMEOUT_MAX):
                raise OperationError(space.w_ValueError, space.wrap(
                        "timeout value is too large"))
            microseconds = r_longlong(timeout)
        mylock = self.lock
        result = mylock.acquire_timed(microseconds)
        return space.newbool(result)

    def descr_lock_release(self, space):
        """Release the lock, allowing another thread that is blocked waiting for
the lock to acquire the lock.  The lock must be in the locked state,
but it needn't be locked by the same thread that unlocks it."""
        try:
            self.lock.release()
        except thread.error:
            raise wrap_thread_error(space, "release unlocked lock")

    def descr_lock_locked(self, space):
        """Return whether the lock is in the locked state."""
        if self.lock.acquire(False):
            self.lock.release()
            return space.w_False
        else:
            return space.w_True

    def descr__enter__(self, space):
        self.descr_lock_acquire(space)
        return self

    def descr__exit__(self, space, __args__):
        self.descr_lock_release(space)

    def __enter__(self):
        self.descr_lock_acquire(self.space)
        return self

    def __exit__(self, *args):
        self.descr_lock_release(self.space)

Lock.typedef = TypeDef(
    "_thread.lock",
    __doc__ = """\
A lock object is a synchronization primitive.  To create a lock,
call the thread.allocate_lock() function.  Methods are:

acquire() -- lock the lock, possibly blocking until it can be obtained
release() -- unlock of the lock
locked() -- test whether the lock is currently locked

A lock is not owned by the thread that locked it; another thread may
unlock it.  A thread attempting to lock a lock that it has already locked
will block until another thread unlocks it.  Deadlocks may ensue.""",
    acquire = interp2app(Lock.descr_lock_acquire),
    release = interp2app(Lock.descr_lock_release),
    locked  = interp2app(Lock.descr_lock_locked),
    __enter__ = interp2app(Lock.descr__enter__),
    __exit__ = interp2app(Lock.descr__exit__),
    # Obsolete synonyms
    acquire_lock = interp2app(Lock.descr_lock_acquire),
    release_lock = interp2app(Lock.descr_lock_release),
    locked_lock  = interp2app(Lock.descr_lock_locked),
    )


def allocate_lock(space):
    """Create a new lock object.  (allocate() is an obsolete synonym.)
See LockType.__doc__ for information about locks."""
    return space.wrap(Lock(space))


class W_RLock(Wrappable):
    def __init__(self, space):
        self.rlock_count = 0
        self.rlock_owner = 0
        try:
            self.lock = thread.allocate_lock()
        except thread.error:
            raise wrap_thread_error(space, "cannot allocate lock")

    def descr__new__(space, w_subtype):
        self = space.allocate_instance(W_RLock, w_subtype)
        W_RLock.__init__(self, space)
        return space.wrap(self)

    def descr__repr__(self):
        typename = space.type(self).getname(space)
        return space.wrap("<%s owner=%d count=%d>" % (
                typename, self.rlock_owner, self.rlock_count))

    @unwrap_spec(blocking=bool)
    def acquire_w(self, space, blocking=True):
        """Lock the lock.  `blocking` indicates whether we should wait
        for the lock to be available or not.  If `blocking` is False
        and another thread holds the lock, the method will return False
        immediately.  If `blocking` is True and another thread holds
        the lock, the method will wait for the lock to be released,
        take it and then return True.
        (note: the blocking operation is not interruptible.)
        
        In all other cases, the method will return True immediately.
        Precisely, if the current thread already holds the lock, its
        internal counter is simply incremented. If nobody holds the lock,
        the lock is taken and its internal counter initialized to 1."""
        tid = thread.get_ident()
        if self.rlock_count > 0 and tid == self.rlock_owner:
            try:
                self.rlock_count = ovfcheck(self.rlock_count + 1)
            except OverflowError:
                raise OperationError(space.w_OverflowError, space.wrap(
                        'internal lock count overflowed'))
            return space.w_True

        r = True
        if self.rlock_count > 0 or not self.lock.acquire(False):
            if not blocking:
                return space.w_False
            r = self.lock.acquire(True)
        if r:
            assert self.rlock_count == 0
            self.rlock_owner = tid
            self.rlock_count = 1

        return space.wrap(r)
            

    def release_w(self, space):
        """Release the lock, allowing another thread that is blocked waiting for
        the lock to acquire the lock.  The lock must be in the locked state,
        and must be locked by the same thread that unlocks it; otherwise a
        `RuntimeError` is raised.
        
        Do note that if the lock was acquire()d several times in a row by the
        current thread, release() needs to be called as many times for the lock
        to be available for other threads."""
        tid = thread.get_ident()
        if self.rlock_count == 0 or self.rlock_owner != tid:
            raise OperationError(space.w_RuntimeError, space.wrap(
                    "cannot release un-acquired lock"))
        self.rlock_count -= 1
        if self.rlock_count == 0:
            self.rlock_owner == 0
            self.lock.release()

    def is_owned_w(self, space):
        """For internal use by `threading.Condition`."""
        tid = thread.get_ident()
        if self.rlock_count > 0 and self.rlock_owner == tid:
            return space.w_True
        else:
            return space.w_False

    @unwrap_spec(count=int, owner=int)
    def acquire_restore_w(self, space, count, owner):
        """For internal use by `threading.Condition`."""
        r = True
        if not self.lock.acquire(False):
            r = self.lock.acquire(True)
        if not r:
            raise wrap_thread_error(space, "coult not acquire lock")
        assert self.rlock_count == 0
        self.rlock_owner = owner
        self.rlock_count = count

    def release_save_w(self, space):
        """For internal use by `threading.Condition`."""
        count, self.rlock_count = self.rlock_count, 0
        owner, self.rlock_owner = self.rlock_owner, 0
        return space.newtuple([space.wrap(count), space.wrap(owner)])

    def descr__enter__(self, space):
        self.acquire_w(space)
        return self

    def descr__exit__(self, space, *args):
        self.release_w(space)

W_RLock.typedef = TypeDef(
    "_thread.RLock",
    __new__ = interp2app(W_RLock.descr__new__.im_func),
    acquire = interp2app(W_RLock.acquire_w),
    release = interp2app(W_RLock.release_w),
    _is_owned = interp2app(W_RLock.is_owned_w),
    _acquire_restore = interp2app(W_RLock.acquire_restore_w),
    _release_save = interp2app(W_RLock.release_save_w),
    __enter__ = interp2app(W_RLock.descr__enter__),
    __exit__ = interp2app(W_RLock.descr__exit__),
    __weakref__ = make_weakref_descr(W_RLock),
    )
