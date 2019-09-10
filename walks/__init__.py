# -*- coding: utf-8 -*-
"""
Purpose
=======

Walks is a collection of methods related to random walks and their analysis.

The following functionalities are directly provided on module-level.


Subpackages
===========

.. autosummary::
    simulation
    output
    plot
    random
    integrator


Classes
=======

Simulation
^^^^^^^^^^

Class for random walk simulations.

.. currentmodule:: walks.simulation

.. autosummary::
   Simulation


MasterRNG
^^^^^^^^^

Class for random number generation.

.. currentmodule:: walks.random

.. autosummary::
   MasterRNG


Output
^^^^^^

Class for righting the results from the  simulations to file.

.. currentmodule:: walks.output

.. autosummary::
   Memory
   Pickle
   Output


Functions
=========

plot
^^^^

Methods for plotting the random walk results.

.. currentmodule:: walks.plot

.. autosummary::
   walks
   video


integrator
^^^^^^^^^^

Methods for integrating the equations of motion.

.. currentmodule:: walks.integrator

.. autosummary::
   euler_maruyama
"""
from __future__ import absolute_import

from walks._version import __version__
from walks.random import MasterRNG
from walks.integrator import euler_maruyama
from walks.simulation import Simulation
from walks.output import Memory, Pickle

# from walks import plot

__all__ = ["__version__"]
__all__ += ["Simulation", "output", "MasterRNG"]
