try:
    import _stacklet
except ImportError:
    import py; py.test.skip("to run on top of a translated pypy-c")

import random

# ____________________________________________________________

STATUS_MAX = 50000
MAIN_RECURSION_MAX = 50
SUB_RECURSION_MAX = 20

def set_fast_mode():
    global STATUS_MAX, MAIN_RECURSION_MAX, SUB_RECURSION_MAX
    STATUS_MAX = 100
    MAIN_RECURSION_MAX = 7
    SUB_RECURSION_MAX = 3

# ____________________________________________________________

class Task:
    def __init__(self, n):
        self.n = n
        self.h = None

    def withdepth(self, d):
        if d > 0:
            res = self.withdepth(d-1)
        else:
            res = 0
            n = random.randrange(10)
            if n == self.n or (Task.status >= STATUS_MAX and
                               not Task.tasks[n].h):
                return 1

            print "status == %d, self.n = %d" % (Task.status, self.n)
            assert not self.h
            assert Task.nextstep == -1
            Task.status += 1
            Task.nextstep = Task.status
            Task.comefrom = self.n
            Task.gointo = n
            task = Task.tasks[n]
            if not task.h:
                # start a new stacklet
                print "NEW", n
                h = _stacklet.newstacklet(variousstackdepths_callback, n)
            else:
                # switch to this stacklet
                print "switch to", n
                h = task.h
                task.h = None
                h = h.switch()

            print "back in self.n = %d, coming from %d" % (self.n,
                                                           Task.comefrom)
            assert Task.nextstep == Task.status
            Task.nextstep = -1
            assert Task.gointo == self.n
            assert Task.comefrom != self.n
            assert self.h is None
            if Task.comefrom != -42:
                assert 0 <= Task.comefrom < 10
                task = Task.tasks[Task.comefrom]
                print "setting %r.h = %r" % (task, h)
                assert task.h is None
                task.h = h
            else:
                assert h is None
            Task.comefrom = -1
            Task.gointo = -1
        assert (res & (res-1)) == 0
        return res

def variousstackdepths_callback(h, arg):
    assert Task.nextstep == Task.status
    Task.nextstep = -1
    assert arg == Task.gointo
    self = Task.tasks[arg]
    assert self.n == Task.gointo
    assert self.h is None
    assert 0 <= Task.comefrom < 10
    task = Task.tasks[Task.comefrom]
    print "initializing %r.h = %r" % (task, h)
    assert task.h is None
    assert type(h) is _stacklet.Stacklet
    task.h = h
    Task.comefrom = -1
    Task.gointo = -1

    while self.withdepth(random.randrange(SUB_RECURSION_MAX)) == 0:
        pass

    assert self.h is None
    while 1:
        n = random.randrange(10)
        h = Task.tasks[n].h
        if h is not None:
            break

    Task.tasks[n].h = None
    Task.comefrom = -42
    Task.gointo = n
    assert Task.nextstep == -1
    Task.status += 1
    Task.nextstep = Task.status
    print "LEAVING %d to go to %d" % (self.n, n)
    return h

def any_alive():
    return any([task.h is not None for task in Task.tasks])

def test_various_depths():
    Task.tasks = [Task(i) for i in range(10)]
    Task.nextstep = -1
    Task.comefrom = -1
    Task.status = 0
    while Task.status < STATUS_MAX or any_alive():
        Task.tasks[0].withdepth(random.randrange(MAIN_RECURSION_MAX))

# ____________________________________________________________

if __name__ == '__main__':
    test_various_depths()
