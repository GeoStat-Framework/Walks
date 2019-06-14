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
    plot


Classes
=======

Simulation
^^^^^^^^^^

Class for random walk simulations.

.. currentmodule:: walks.simulation

.. autosummary::
   Simulation
"""
from __future__ import absolute_import

from walks._version import __version__
from walks.random import MasterRNG
from walks.integrator import euler_maruyama
from walks.simulation import Simulation
from walks.output import Memory, Pickle
from walks import plot

__all__ = ['__version__']
__all__ += ['Simulation']
