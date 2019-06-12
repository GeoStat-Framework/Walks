# -*- coding: utf-8 -*-
"""
Walk module providing a class for random walk simulations.

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
from walks.output import Pickle


OUTPUT = {
    'pickle': Pickle,
    'NetCDF': Pickle,
    'VTK': Pickle,
}


class Simulation(object):
    """Perform particle experiments.

    Parameters
    ----------
    field : :any:`SRF`
        A velocity field
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
        output='pickle',
        filename='walks.p',
        **field_kwargs
    ):
        self.dim = dim
        self.field = field
        self.D = D
        self.T = T
        self.dt = dt
        self.nsave = nsave
        # will be initialised by self.initial_condition()
        self.pos = None
        self.N = None

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
            pos = pos[:,np.newaxis]
        self.pos = np.repeat(pos, distribution, axis=1)
        self.N = self.pos.shape[1]

    def __call__(self, seed=None):
        """Simulate the random walk.
        
        Parameters
        ----------
        seed : :class:`int`, optional
            RNG seed
        """
        if self.pos is None:
            raise RuntimeError('Set the initial conditions before simulating.')

        rngs = self._create_rng_streams(self.dim, seed)
        jumps = np.empty_like(self.pos)

        # write initial conditions to file
        self.output.write_timestep(0., self.pos)
        for timestep, t in enumerate(np.arange(0., self.T, self.dt)):
            jumps[:,:] = [rngs[d].standard_normal(self.N) for d in range(self.dim)]
            drift = self.field(self.pos, **self.field_kwargs)
            euler_maruyama(self.pos, drift, jumps, self.D, self.dt)
            if timestep % self.nsave == 0:
                self.output.write_timestep(t, self.pos)

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
