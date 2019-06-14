#!python
#cython: language_level=2
# distutils: language = c++
# -*- coding: utf-8 -*-
"""
The random walk integrators, implemented in Cython.
"""
from __future__ import division, absolute_import, print_function

import numpy as np

cimport cython
from cython.parallel import prange
from libc.math cimport sqrt
cimport numpy as np


#cdef extern from '<random>' namespace 'std':
#    cdef cppclass default_random_engine:
#        default_random_engine()
#        default_random_engine(unsigned int seed)
#
#    cdef cppclass normal_distribution[T]:
#        normal_distribution()
#        normal_distribution(T mean, T stddev)
#        T operator()(default_random_engine rng)


DTYPE = np.double
ctypedef np.double_t DTYPE_t


#cdef default_random_engine rng = default_random_engine(4738845)
#cdef normal_distribution[double] eta = normal_distribution[double](0., 1.)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def euler_maruyama(
    double[:,:] pos,
    double[:,:] drift,
    double[:,:] jumps,
    double[:] D,
    double dt
    ):
    cdef int i, d, dim, N

    dim = pos.shape[0]
    N = pos.shape[1]

    for d in range(dim):
        for i in range(N):
            pos[d,i] = pos[d,i] + drift[d,i] * dt + sqrt(2.*D[d] * dt) * jumps[d,i]
