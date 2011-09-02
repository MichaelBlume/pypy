from pypy.jit.metainterp.resoperation import rop
from pypy.jit.metainterp.history import ConstInt

class HeapCache(object):
    def __init__(self):
        self.reset()

    def reset(self):
        # contains boxes where the class is already known
        self.known_class_boxes = {}
        # contains frame boxes that are not virtualizables
        self.nonstandard_virtualizables = {}
        # heap cache
        # maps descrs to (from_box, to_box) tuples
        self.heap_cache = {}
        # heap array cache
        # maps descrs to {index: (from_box, to_box)} dicts
        self.heap_array_cache = {}

    def invalidate_caches(self, opnum, descr):
        if opnum == rop.SETFIELD_GC:
            return
        if opnum == rop.SETARRAYITEM_GC:
            return
        if rop._NOSIDEEFFECT_FIRST <= opnum <= rop._NOSIDEEFFECT_LAST:
            return
        if opnum == rop.CALL:
            effectinfo = descr.get_extra_info()
            ef = effectinfo.extraeffect
            if ef == effectinfo.EF_LOOPINVARIANT or \
               ef == effectinfo.EF_ELIDABLE_CANNOT_RAISE or \
               ef == effectinfo.EF_ELIDABLE_CAN_RAISE:
                return
        self.heap_cache.clear()
        self.heap_array_cache.clear()

    def is_class_known(self, box):
        return box in self.known_class_boxes

    def class_now_know(self, box):
        self.known_class_boxes[box] = None

    def is_nonstandard_virtualizable(self, box):
        return box in self.nonstandard_virtualizables

    def nonstandard_virtualizables_now_known(self, box):
        self.nonstandard_virtualizables[box] = None


    def getfield(self, box, descr):
        frombox, tobox = self.heap_cache.get(descr, (None, None))
        if box is frombox:
            return tobox
        return None

    def setfield(self, box, descr, fieldbox):
        self.heap_cache[descr] = (box, fieldbox)

    def getarrayitem(self, box, descr, indexbox):
        if not isinstance(indexbox, ConstInt):
            return
        index = indexbox.getint()
        cache = self.heap_array_cache.get(descr, None)
        if cache:
            frombox, tobox = cache.get(index, (None, None))
            if frombox is box:
                return tobox

    def setarrayitem(self, box, descr, indexbox, valuebox):
        if not isinstance(indexbox, ConstInt):
            cache = self.heap_array_cache.get(descr, None)
            if cache is not None:
                cache.clear()
            return
        cache = self.heap_array_cache.setdefault(descr, {})
        index = indexbox.getint()
        cache[index] = box, valuebox

    def replace_box(self, oldbox, newbox):
        for descr, (frombox, tobox) in self.heap_cache.iteritems():
            change = False
            if frombox is oldbox:
                change = True
                frombox = newbox
            if tobox is oldbox:
                change = True
                tobox = newbox
            if change:
                self.heap_cache[descr] = frombox, tobox
        # XXX what about self.heap_array_cache?
