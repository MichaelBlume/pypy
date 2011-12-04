from pypy.rlib.rrandom import Random, N, r_uint
from pypy.rlib.rarithmetic import intmask
import _random

# the numbers were created by using CPython's _randommodule.c

def test_init_from_zero():
    rnd = Random(0)
    assert rnd.state[:14] == [0, 1, 1812433255, 1900727105, 1208447044,
            2481403966, 4042607538, 337614300, 3232553940,
            1018809052, 3202401494, 1775180719, 3192392114, 594215549]

def test_init_from_seed():
    rnd = Random(1000)
    assert rnd.state[:14] == [1000, 4252021385, 1724402292, 571538732,
            73690720, 4283493349, 2222270404, 2464917285, 427676011,
            1101193792, 2887716015, 3670250828, 1664757559, 1538426459]

def test_numbers():
    rnd = Random(1000)
    nums = [rnd.genrand32() for i in range(14)]
    assert nums == [2807145907, 882709079, 493951047, 2621574848, 4081433851,
            44058974, 2070996316, 1549632257, 3747249597, 3650674304,
            911961945, 58396205, 174846504, 1478498153]

def test_init_by_array():
    rnd = Random()
    rnd.init_by_array([r_uint(n) for n in [1, 2, 3, 4]])
    assert rnd.state[:14] == [2147483648, 1269538435, 699006892, 381364451,
            172015551, 3237099449, 3609464087, 2187366456, 654585064,
            2665903765, 3735624613, 1241943673, 2038528247, 3774211972]
    # try arrays of various sizes to test for corner cases
    for size in [N, N - 1, N + 1, N // 2, 2 * N]:
        rnd.init_by_array([r_uint(n) for n in range(N)])

def test_jumpahead():
    rnd = Random()
    rnd.state = [r_uint(0)] * N
    rnd.state[0] = r_uint(1)
    cpyrandom = _random.Random()
    cpyrandom.setstate(tuple([int(s) for s in rnd.state] + [rnd.index]))
    rnd.jumpahead(100)
    cpyrandom.jumpahead(100)
    assert tuple(rnd.state) + (rnd.index, ) == cpyrandom.getstate()

def test_translate():
    from pypy.translator.interactive import Translation
    def f(x, y):
        rnd = Random(x)
        rnd.init_by_array([x, y])
        rnd.jumpahead(intmask(y))
        return rnd.genrand32(), rnd.random()
    t = Translation(f)
    fc = t.compile_c([r_uint, r_uint])
    assert fc(r_uint(1), r_uint(2)) == f(r_uint(1), r_uint(2))
