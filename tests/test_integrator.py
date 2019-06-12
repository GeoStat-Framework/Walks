#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, absolute_import, print_function

import numpy as np
import unittest

from walks import integrator


class TestIntegrator(unittest.TestCase):
    def setUp(self):
       self.pos_2d = np.array(((0., 0., 0.), (0., 0., 0.)))
       self.drift_2d = np.array(((1., 0., 0.), (0., 1., 0.)))
       self.drift_2d = np.array(((1., 0., 0.), (0., 1., 0.)))
       self.jumps_2d = np.array(((.0, .1, -.1), (.1, -.1, .0)))
       self.jumps_2d = np.array(((.0, .1, -.1), (.1, -.1, .0)))
       self.D_2d = np.atleast_1d(np.array((0.01, 0.01)))
       self.dt = 1.

    def test_euler_maruyama(self):
        integrator.euler_maruyama(
            self.pos_2d,
            self.drift_2d,
            self.jumps_2d,
            self.D_2d,
            self.dt,
        )
        integrator.euler_maruyama(
            self.pos_2d,
            self.drift_2d,
            self.jumps_2d,
            self.D_2d,
            self.dt,
        )

        self.assertAlmostEqual(self.pos_2d[0,0], 2*self.drift_2d[0,0])

        diffusion = np.sqrt(2.*self.D_2d[0]*self.dt)*self.jumps_2d[0,1]
        self.assertAlmostEqual(self.pos_2d[0,1], 2*diffusion)


if __name__ == '__main__':
    unittest.main()
