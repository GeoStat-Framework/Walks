# -*- coding: utf-8 -*-
"""
Module providing a class for random walk simulations.

.. currentmodule:: walks.simulation

The following classes are provided

.. autosummary::
   Simulation
"""
# pylint: disable=C0103
from __future__ import division, absolute_import, print_function

import numpy as np

from walks.random import MasterRNG
from walks.integrator import euler_maruyama
from walks.output import Memory, Pickle

__all__ = ["Simulation"]

OUTPUT = {"memory": Memory, "pickle": Pickle, "NetCDF": Pickle, "VTK": Pickle}


class Sources(object):
    def __init__(self):
        self.t = []
        self.pos = []
        # TODO use getters to automatically increase idx
        self.idx = 0


class Simulation(object):
    """Perform particle experiments.

    Parameters
    ----------
    field :
        A callable object, which takes a position tuple and returns a tuple
    D : :class:`np.ndarray`
        the diffusion tensor
    T : :class:`float`
        Simulation time
    dt : :class:`float`
        Time step
    nsave : :class:`int`, optional
        write output every nsave'th step
    """

    def __init__(
        self,
        dim,
        field,
        D,
        T,
        dt,
        nsave=1,
        output="memory",
        filename="walks.p",
        **field_kwargs
    ):
        self.dim = dim
        self.field = field
        self.D = D
        self.T = T
        self.dt = dt
        self.nsave = nsave
        self.pos = np.asarray(([None] * self.dim))[:, np.newaxis]
        self.N = 0
        self.sources = Sources()
        # add one timepoint after max. simulation time for the pops to not through
        # an exception when all sources have been added
        self.sources.t = [self.T + self.dt]

        if output in OUTPUT:
            out = OUTPUT[output]
            self.output = out(filename)

        self.field_kwargs = field_kwargs

    def initial_condition(self, pos, distribution=1):
        """Initialise the initial particle positions.

        Parameters
        ----------
            pos : :any:`numpy.ndarray`
                Initial positions of the particles, given as a tuple of
                positions
            distribution : :any:`numpy.ndarray` or :class:`int`, optional
                number of walkers at each pos
        """
        pos = np.array(pos)
        # np.repeat needs this for axis=1
        if len(pos.shape) == 1:
            pos = pos[:, np.newaxis]
        self.pos = np.repeat(pos, distribution, axis=1)
        self.N = self.pos.shape[1]

    def add_sources(self, times, pos, distribution=1):
        if not isinstance(times, list):
            times = [times]
        if not isinstance(pos, list):
            pos = [pos]
        self.sources.t = list(times)
        # add one timepoint after max. simulation time for the pops to not through
        # an exception when all sources have been added
        self.sources.t.append([self.T + self.dt])
        self.sources.pos = np.atleast_1d(pos)
        distribution = np.atleast_1d(distribution)
        if len(distribution) == 1:
            distribution = np.repeat(distribution, len(times))
        self.sources.distribution = list(distribution)

    def __call__(self, seed=None):
        """Simulate the random walk.
        
        Parameters
        ----------
        seed : :class:`int`, optional
            RNG seed
        """

        rngs = self._create_rng_streams(self.dim, seed)
        self.jumps = np.empty_like(self.pos)

        print("Starting simulation with {} walkers.".format(self.N))

        # write initial conditions to file
        self.output.write_timestep(0.0, self.pos)
        for timestep, t in enumerate(np.arange(0.0, self.T, self.dt)):
            if self.N > 0:
                self.jumps[:, :] = [
                    rngs[d].standard_normal(self.N) for d in range(self.dim)
                ]
                drift = self.field(self.pos, **self.field_kwargs)
                euler_maruyama(self.pos, drift, self.jumps, self.D, self.dt)
            if t <= self.sources.t[self.sources.idx] < t + self.dt:
                self._apply_sources()
            if timestep % self.nsave == 0:
                self.output.write_timestep(t, self.pos)
        print("Simulation ended with {} walkers.".format(self.N))

    def _apply_sources(self):
        source_pos = self.sources.pos[:, self.sources.idx]
        if len(source_pos.shape) == 1:
            source_pos = source_pos[:, np.newaxis]
        distribution = self.sources.distribution[self.sources.idx]
        source_pos = np.repeat(source_pos, distribution, axis=1)
        self.pos = np.hstack((self.pos, source_pos))
        self.N = self.pos.shape[1]
        self.jumps = np.empty_like(self.pos)
        self.sources.idx += 1

    def _create_rng_streams(self, dim, seed):
        """Create a RNG stream for each spatial dimension.
        
        Parameters
        ----------
        dim : :class:`int`
            spatial dimension
        seed : :class`int`
            master seed
        """
        master_rng = MasterRNG(seed)
        streams = [np.random.RandomState(master_rng()) for d in range(dim)]
        return streams

    @property
    def mean_pos(self):
        """:any:`numpy.ndarray`: mean postition of all walkers."""
        return np.mean(self.pos, axis=1)
