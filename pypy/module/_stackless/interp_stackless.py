from pypy.interpreter.baseobjspace import Wrappable
from pypy.interpreter.typedef import TypeDef
from pypy.interpreter.gateway import interp2app
import os


class tasklet(Wrappable):

    def __init__(self, space):
        self.space = space
        self.flags = 0
        self.state = None

    def descr_method__new__(space, w_subtype):
        t = space.allocate_instance(tasklet, w_subtype)
        tasklet.__init__(t, space)
        return space.wrap(t)

    def w_demo(self):
        output("42")

tasklet.typedef = TypeDef("tasklet",
    __new__ = interp2app(tasklet.descr_method__new__.im_func),
    demo = interp2app(tasklet.w_demo),
)

def output(stuff):
    os.write(2, stuff + '\n')
