# -*- coding: utf-8 -*-
"""
Tools for random number generation.

.. currentmodule:: walks.random

The following classes are provided

.. autosummary::
   MasterRNG
"""
from __future__ import division, absolute_import, print_function

import numpy.random as rand

__all__ = ["MasterRNG"]


class MasterRNG(object):
    """Master random number generator for generating seeds.

    Parameters
    ----------
    seed : :class:`int` or :any:`None`, optional
        The seed of the master RNG, if ``None``,
        a random seed is used. Default: ``None``

    """

    def __init__(self, seed):
        self._seed = seed
        self._master_rng_fct = rand.RandomState(seed)
        self._master_rng = lambda: self._master_rng_fct.randint(1, 2 ** 16)

    def __call__(self):
        """Return a random seed."""
        return self._master_rng()

    @property
    def seed(self):
        """:class:`int`: Seed of the master RNG.

        The setter property not only saves the new seed, but also creates
        a new master RNG function with the new seed.
        """
        return self._seed

    def __str__(self):
        """Return String representation."""
        return self.__repr__()

    def __repr__(self):
        """Return String representation."""
        return "RNG(seed={})".format(self.seed)
