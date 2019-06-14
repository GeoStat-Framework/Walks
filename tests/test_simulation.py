#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, absolute_import, print_function

import numpy as np
import unittest

from gstools import SRF, Gaussian

from walks import Simulation


class TestSimulation(unittest.TestCase):
    def srf(self, pos):
        """SRF emulation"""
        return self.drift_2d

    def setUp(self):
        self.D_2d = np.atleast_1d(np.array((0.01, 0.01)))
        self.T = 10.
        self.dt = 1.
        self.nsave = 1
        self.pos_2d = np.array(((0., 0., 0., 0.), (0., 1., 2., 3.)))
        self.distribution_2d = 2
        self.N = self.distribution_2d * self.pos_2d.shape[1]
        self.drift_2d = np.repeat(np.array((1., 0.))[:,np.newaxis], self.N, axis=1)
        self.jumps_2d = np.array(((.0, .1, -.1), (.1, -.1, .0)))
        self.dt = 1.

    def test_drift(self):
        D0 = np.array((0., 0.))
        T = 3
        sim = Simulation(2, self.srf, D0, T, self.dt)
        sim.initial_condition(self.pos_2d, self.distribution_2d)
        pos_orig = np.copy(sim.pos)
        sim()
        # total_drift is only correct for first and last entry
        # (see Simulation.initial_condition)
        total_drift = 3 * self.drift_2d * self.dt
        for d in range(sim.dim):
            self.assertAlmostEqual(sim.pos[d,0], pos_orig[d,0] + total_drift[d,0])
            self.assertAlmostEqual(sim.pos[d,-1], pos_orig[d,-1] + total_drift[d,-1])

    def test_diffusion(self):
        T = 1000
        dim = 2
        pos = np.zeros((dim, 10000))
        N = pos.shape[1]
        self.drift_2d = np.repeat(np.array((0., 0.))[:,np.newaxis], N, axis=1)
        sim = Simulation(dim, self.srf, self.D_2d, T, self.dt)
        sim.initial_condition(pos)
        sim()
        for d in range(dim):
            self.assertAlmostEqual(sim.mean_pos[d], 0., places=1)

    def test_initial_condition(self):
        sim = Simulation(2, self.srf, self.D_2d, self.T, self.dt)
        sim.initial_condition(self.pos_2d, self.distribution_2d)
        self.assertEqual(len(sim.pos[0,:]), self.N)
        self.assertEqual(sim.pos[1,-1], self.pos_2d[1,-1])
        self.assertEqual(sim.pos[1,-2], self.pos_2d[1,-1])

        sim = Simulation(2, self.srf, self.D_2d, self.T, self.dt)
        sim.initial_condition(self.pos_2d, 1)
        for d in range(sim.dim):
            for i in range(sim.N):
                self.assertEqual(sim.pos[d,i], self.pos_2d[d,i])

        sim.initial_condition(self.pos_2d, 2)
        self.assertEqual(len(sim.pos[0,:]), self.N)
        self.assertEqual(sim.pos[1,-1], self.pos_2d[1,-1])
        self.assertEqual(sim.pos[1,-2], self.pos_2d[1,-1])

        pos = [0., 0.]
        sim = Simulation(2, self.srf, self.D_2d, self.T, self.dt)
        sim.initial_condition(pos, 100)

    def test_srf(self):
        D0 = np.array((0., 0.))
        cov_model = Gaussian(dim=2, var=.01, len_scale=10.)
        srf = SRF(cov_model, generator='VectorField', seed=5747387)

        sim = Simulation(2, srf, D0, self.T, self.dt)
        sim.initial_condition(self.pos_2d, self.distribution_2d)
        mean_drift = srf.generator.mean_u * (self.T / self.dt)
        sim()
        self.assertAlmostEqual(sim.pos[0,0], mean_drift, places=0)
        self.assertAlmostEqual(sim.pos[1,1], 0., places=0)
        self.assertAlmostEqual(sim.pos[0,-1], mean_drift, places=0)


if __name__ == '__main__':
    unittest.main()
