#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

import asyncio
import h5py
import numpy as np
from unittest import TestCase
from numpy.testing import assert_array_equal

from amfe.io.tools import amfe_dir, check_dir
from amfe.solver import AmfeSolution, solve_async


class DummySolver:
    def __init__(self):
        self.ndof = 100
        self.strains = None

    def solve(self, callback, callbackargs=(), t0=0.0, tend=1.0, dt=0.01):
        t, q, dq, ddq, strains, stresses = self._initialize(t0, tend, dt, self.ndof)

        t = np.arange(t0, tend, dt)

        for i, t_current in enumerate(t):
            callback(t_current, q[i, :], dq[i, :], ddq[i, :], strains[i, :, :], stresses[i, :, :], *callbackargs)

    async def solve_async(self, callback, callbackargs=(), t0=0.0, tend=1, dt=0.01):
        t, q, dq, ddq, strains, stresses = self._initialize(t0, tend, dt, self.ndof)

        t = np.arange(t0, tend, dt)

        for i, t_current in enumerate(t):
            await callback(t_current, q[i, :], dq[i, :], ddq[i, :], strains[i, :, :], stresses[i, :, :], *callbackargs)

    def _initialize(self, t0, tend, dt, ndof):
        t = np.arange(t0, tend, dt)
        q = np.array([np.arange(0, ndof)*scale for scale in t])
        dq = q.copy()
        ddq = q.copy()
        if self.strains is None:
            self.strains = np.array([np.random.rand(ndof, 6) * scale for scale in t])
        strains = self.strains
        stresses = strains.copy()
        return t, q, dq, ddq, strains, stresses


class AmfeSolutionTest(TestCase):
    def setUp(self):
        self.solver = DummySolver()
        return

    def tearDown(self):
        return

    def test_amfe_solution(self):
        # Only q
        solution = AmfeSolution()
        q1 = np.arange(0, 60, dtype=float)
        q2 = np.arange(10, 70, dtype=float)
        t1 = 0.1
        t2 = 0.5

        solution.write_timestep(t1, q1)
        solution.write_timestep(t2, q2)

        assert_array_equal(solution.q[0], q1)
        assert_array_equal(solution.q[1], q2)
        self.assertEqual(solution.t[0], t1)
        self.assertEqual(solution.t[1], t2)
        self.assertTrue(len(solution.t) == len(solution.q))
        self.assertEqual(len(solution.t), 2)

        # q and dq
        solution = AmfeSolution()
        dq1 = np.arange(20, 80, dtype=float)
        dq2 = np.arange(30, 90, dtype=float)

        solution.write_timestep(t1, q1, dq1)
        solution.write_timestep(t2, q2, dq2)

        assert_array_equal(solution.q[0], q1)
        assert_array_equal(solution.q[1], q2)
        assert_array_equal(solution.dq[0], dq1)
        assert_array_equal(solution.dq[1], dq2)
        self.assertEqual(solution.t[0], t1)
        self.assertEqual(solution.t[1], t2)
        self.assertTrue(len(solution.t) == len(solution.q))
        self.assertTrue(len(solution.t) == len(solution.dq))
        self.assertEqual(len(solution.t), 2)

        # q, dq and ddq
        solution = AmfeSolution()
        ddq1 = np.arange(40, 100, dtype=float)
        ddq2 = np.arange(50, 110, dtype=float)

        solution.write_timestep(t1, q1, dq1, ddq1)
        solution.write_timestep(t2, q2, dq2, ddq2)

        assert_array_equal(solution.q[0], q1)
        assert_array_equal(solution.q[1], q2)
        assert_array_equal(solution.dq[0], dq1)
        assert_array_equal(solution.dq[1], dq2)
        assert_array_equal(solution.ddq[0], ddq1)
        assert_array_equal(solution.ddq[1], ddq2)
        self.assertEqual(solution.t[0], t1)
        self.assertEqual(solution.t[1], t2)
        self.assertTrue(len(solution.t) == len(solution.q))
        self.assertTrue(len(solution.t) == len(solution.dq))
        self.assertTrue(len(solution.t) == len(solution.ddq))
        self.assertEqual(len(solution.t), 2)

        # q and ddq
        solution = AmfeSolution()

        solution.write_timestep(t1, q1, ddq=ddq1)
        solution.write_timestep(t2, q2, ddq=ddq2)

        assert_array_equal(solution.q[0], q1)
        assert_array_equal(solution.q[1], q2)
        assert_array_equal(solution.ddq[0], ddq1)
        assert_array_equal(solution.ddq[1], ddq2)
        self.assertEqual(solution.t[0], t1)
        self.assertEqual(solution.t[1], t2)
        self.assertTrue(len(solution.t) == len(solution.q))
        self.assertTrue(len(solution.t) == len(solution.ddq))
        self.assertEqual(len(solution.t), 2)
        self.assertIsNone(solution.dq[0])
        self.assertIsNone(solution.dq[1])

        # q and strains
        solution = AmfeSolution()
        strains1 = np.random.rand(60, 6)
        strains2 = np.random.rand(60, 6)

        solution.write_timestep(t1, q1, strain=strains1)
        solution.write_timestep(t2, q2, strain=strains2)
        self.assertEqual(solution.t[0], t1)
        self.assertEqual(solution.t[1], t2)
        assert_array_equal(solution.q[0], q1)
        assert_array_equal(solution.q[1], q2)
        assert_array_equal(solution.strain[0], strains1)
        assert_array_equal(solution.strain[1], strains2)
        self.assertIsNone(solution.stress[0])
        self.assertIsNone(solution.stress[1])

        # q, strains and stresses
        solution = AmfeSolution()
        strains1 = np.random.rand(60, 6)
        strains2 = np.random.rand(60, 6)
        stresses1 = np.random.rand(60, 6)
        stresses2 = np.random.rand(60, 6)

        solution.write_timestep(t1, q1, strain=strains1, stress=stresses1)
        solution.write_timestep(t2, q2, strain=strains2, stress=stresses2)
        self.assertEqual(solution.t[0], t1)
        self.assertEqual(solution.t[1], t2)
        assert_array_equal(solution.q[0], q1)
        assert_array_equal(solution.q[1], q2)
        assert_array_equal(solution.strain[0], strains1)
        assert_array_equal(solution.strain[1], strains2)
        assert_array_equal(solution.stress[0], stresses1)
        assert_array_equal(solution.stress[1], stresses2)
        return
