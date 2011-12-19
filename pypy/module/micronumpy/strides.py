
def calculate_slice_strides(start, strides, backstrides, chunks):
    rstrides = []
    rbackstrides = []
    rstart = start
    i = -1
    for i, (start_, stop, step, lgt) in enumerate(chunks):
        if step != 0:
            rstrides.append(strides[i] * step)
            rbackstrides.append(strides[i] * (lgt - 1) * step)
        rstart += strides[i] * start_
    # add a reminder
    s = i + 1
    assert s >= 0
    rstrides += strides[s:]
    rbackstrides += backstrides[s:]
    return rstart, rstrides, rbackstrides

def calculate_broadcast_strides(strides, backstrides, orig_shape, res_shape):
    rstrides = []
    rbackstrides = []
    for i in range(len(orig_shape)):
        if orig_shape[i] == 1:
            rstrides.append(0)
            rbackstrides.append(0)
        else:
            rstrides.append(strides[i])
            rbackstrides.append(backstrides[i])
    rstrides = [0] * (len(res_shape) - len(orig_shape)) + rstrides
    rbackstrides = [0] * (len(res_shape) - len(orig_shape)) + rbackstrides
    return rstrides, rbackstrides
