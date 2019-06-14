# -*- coding: utf-8 -*-
"""
Walk module providing output functionality.

.. currentmodule:: walks.output

The following classes are provided

.. autosummary::
   Pickle
"""
# pylint: disable=C0103
from __future__ import division, absolute_import, print_function

import pickle
import numpy as np


class Output(object):
    def __init__(self, filename):
        """Save the walks for the afterworld.

        Parameters
        ----------
            filename : :class:`str`
                the name of the output file
        """
        self.filename = filename
        self._file = open(self.filename, 'wb')

    def __del__(self):
        self._file.close()

    def write_timestep(self, time, pos):
        """Write the positions of the walkers to file.

        Parameters
        ----------
            time : :class:`float`
                the current simulation time
            pos : :any:`numpy.ndarray`
                Positions of the particles, given as a tuple of
                positions

        """
        pass

    def load(self):
        """Return the saved values.

        Returns
        -------
        :any:`numpy.ndarray`
            Simulation time of saved time steps
        :any:`numpy.ndarray`
            Position of walkers
        """
        pass

class Memory(object):
    def __init__(self, filename):
        self.time = []
        self.pos = []
        self.N = []

    def write_timestep(self, time, pos):
        """Save the positions of the walkers to a Python list.

        Parameters
        ----------
            time : :class:`float`
                the current simulation time
            pos : :any:`numpy.ndarray`
                Positions of the particles, given as a tuple of
                positions

        """
        self.time.append(time)
        self.pos.append(pos.copy())
        self.N.append(pos.shape[1])

    def load(self):
        """Return the saved values.

        Returns
        -------
        :any:`numpy.ndarray`
            Simulation time of saved time steps
        :any:`numpy.ndarray`
            Position of walkers
        """
        time = np.array(self.time)
        timesteps = len(time)

        N = np.array(self.N)
        N_max = N.max()

        dim = self.pos[0].shape[0]

        pos = np.empty((timesteps, dim, N_max))
        pos[:] = np.nan

        for i in range(len(self.pos)):
            n = self.pos[i].shape[1]
            pos[i,:,0:n] = self.pos[i]

        pos = np.ma.masked_invalid(pos)

        return time, pos

class Pickle(Output):
    def __init__(self, filename):
        super().__init__(filename)

    def write_timestep(self, time, pos):
        """Write the positions of the walkers to a pickle file.

        The output is a dictionary with keywords

            * time
            * pos

        Parameters
        ----------
            time : :class:`float`
                the current simulation time
            pos : :any:`numpy.ndarray`
                Positions of the particles, given as a tuple of
                positions
        """
        d = {'time': time, 'pos': pos, 'N': pos.shape[1]}
        pickle.dump(d, self._file)

    def load(self):
        """Load the pickle file.

        Returns
        -------
        :any:`numpy.ndarray`
            Simulation time of saved time steps
        :any:`numpy.ndarray`
            Position of walkers
        """
        self._file.close()
        self._file = open(self.filename, 'rb')
        time = []
        pos = []
        N = []
        while True:
            try:
                d = pickle.load(self._file)
                time.append(d['time'])
                pos.append(d['pos'])
                N.append(d['N'])
            except EOFError:
                break
        time = np.array(time)
        timesteps = len(time)

        N = np.array(N)
        N_max = N.max()

        dim = pos[0].shape[0]

        pos = np.empty((timesteps, dim, N_max))
        pos[:] = np.nan

        for i in range(len(self.pos)):
            n = self.pos[i].shape[1]
            pos[i,:,0:n] = self.pos[i]

        pos = np.ma.masked_invalid(pos)

        return time, pos


class NetCDF(Output):
    def __init__(self, filename):
        super().__init__(filename)
