====================
PyPy 1.9 - Yard Wolf
====================

We're pleased to announce the 1.9 release of PyPy. This release brings mostly
bugfixes, performance improvements, other small improvements and overall
progress on the `numpypy`_ effort.
It also brings an improved situation on windows and OS X.

You can download the PyPy 1.9 release here:

    http://pypy.org/download.html 

.. _`numpypy`: http://pypy.org/numpydonate.html


What is PyPy?
=============

PyPy is a very compliant Python interpreter, almost a drop-in replacement for
CPython 2.7. It's fast (`pypy 1.9 and cpython 2.7.2`_ performance comparison)
due to its integrated tracing JIT compiler.

This release supports x86 machines running Linux 32/64, Mac OS X 64 or
Windows 32.  Windows 64 work is still stalling, we would welcome a volunteer
to handle that.

.. _`pypy 1.9 and cpython 2.7.2`: http://speed.pypy.org


Thanks to our donators
======================

But first of all, we would like to say thank you to all people who
donated some money to one of our four calls:

  * `NumPy in PyPy`_ (got so far $44502 out of $60000)

  * `Py3k (Python 3)`_ (got so far $43563 out of $105000)

  * `Software Transactional Memory`_ (got so far $21791 of $50400)

  * as well as our general PyPy pot.

Thank you all for proving that it is indeed possible for a small team of
(inexpensive) programmers to get funded like that, at least for some
time.  We want to include this thank you in the present release
announcement even though most of the work is not finished yet.  More
precisely, neither Py3k nor STM are ready to make it an official release
yet: people interested in them need to grab and (attempt to) translate
PyPy from the corresponding branches (respectively ``py3k`` and
``stm-thread``).


Highlights
==========

* This release still implements Python 2.7, using the standard library of
  CPython 2.7.2.

* Many bugs were corrected for Windows 32 bit.  This includes new
  functionality to test the validity of file descriptors; and
  correct handling of the calling convensions for ctypes.  (Still not
  much progress on Win64.)

* Improvements in ``cpyext``, our emulator for CPython C extension modules.
  For example PyOpenSSL should now work.

* Sets now have strategies just like dictionaries. This means for example
  that a set containing only ints will be more compact (and faster).

* A lot of progress on various aspects of ``numpypy``.

* The non-x86 backends for the JIT are progressing but are still not
  merged (ARMv7 and PPC64).



XXX should we do something with whatsnew-1.9.txt?
