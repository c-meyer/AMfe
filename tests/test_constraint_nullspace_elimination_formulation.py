"""Test routines for nullspace elimination"""

from unittest import TestCase
import numpy as np
from scipy.sparse import csr_matrix
from numpy.testing import assert_array_equal, assert_allclose

from amfe.constraint.constraint_formulation_nullspace_elimination import NullspaceConstraintFormulation

from .tools import jacobian_finite_difference


class TestNullspaceConstraintFormulation(TestCase):
    def setUp(self):
        M_mat = np.array([[1, -1, 0], [-1, 1.2, -1.5], [0, -1.5, 2]], dtype=np.float64)
        K_mat = np.array([[2, -1, 0], [-1, 2, -1.5], [0, -1.5, 3]], dtype=np.float64)
        D_mat = 0.2 * M_mat + 0.1 * K_mat

        self.M_unconstr = csr_matrix(M_mat)
        self.D_unconstr = csr_matrix(D_mat)
        self.K_unconstr = csr_matrix(K_mat)
        self.f_int_unconstr = np.array([1, 2, 3], dtype=np.float64)
        self.f_ext_unconstr = np.array([3, 4, 5], dtype=float)

        def M(u, du, t):
            return self.M_unconstr

        def h(u, du, t):
            return self.f_int_unconstr

        def p(u, du, t):
            return self.f_ext_unconstr

        def h_q(u, du, t):
            return self.K_unconstr

        def h_dq(u, du, t):
            return self.D_unconstr

        def g_holo(u, t):
            return np.array(u[0], dtype=float, ndmin=1)

        def B_holo(u, t):
            return csr_matrix(np.array([[1, 0, 0]], dtype=float, ndmin=2))

        def c_holo(u, du, t):
            return np.array([0], ndmin=1, dtype=float)

        self.no_of_constraints = 1
        self.no_of_dofs_unconstrained = 3
        self.M_func = M
        self.h_func = h
        self.p_func = p
        self.h_q_func = h_q
        self.h_dq_func = h_dq
        self.g_holo_func = g_holo
        self.B_holo_func = B_holo
        self.c_holo_func = c_holo

        self.formulation = NullspaceConstraintFormulation(self.no_of_dofs_unconstrained, self.M_func, self.h_func,
                                                          self.B_holo_func, self.p_func,
                                                          self.h_q_func, self.h_dq_func,
                                                          g_func=self.g_holo_func, a_func=self.c_holo_func)

    def tearDown(self):
        self.formulation = None

    def test_no_of_dofs_unconstrained(self):
        self.assertEqual(self.formulation.no_of_dofs_unconstrained,
                         self.no_of_dofs_unconstrained)

        self.formulation.no_of_dofs_unconstrained = 5
        self.assertEqual(self.formulation.no_of_dofs_unconstrained,
                         5)

    def test_dimension(self):
        self.assertEqual(self.formulation.dimension,
                         self.no_of_dofs_unconstrained)

    def test_update(self):
        # Just test if update function works
        self.formulation.update()

    def test_recover_u_du_ddu(self):
        x = np.arange(self.formulation.dimension, dtype=float)
        dx = x.copy() + 1.0
        ddx = dx.copy() + 1.0

        u, du, ddu = self.formulation.recover(x, dx, ddx, 5.0)
        assert_array_equal(u, x[:self.no_of_dofs_unconstrained])
        assert_array_equal(du, dx[:self.no_of_dofs_unconstrained])
        assert_array_equal(ddu, ddx[:self.no_of_dofs_unconstrained])

        u = self.formulation.u(x, 2.0)
        du = self.formulation.du(x, dx, 3.0)
        ddu = self.formulation.ddu(x, dx, ddx, 6.0)
        assert_array_equal(u, x[:self.no_of_dofs_unconstrained])
        assert_array_equal(du, dx[:self.no_of_dofs_unconstrained])
        assert_array_equal(ddu, ddx[:self.no_of_dofs_unconstrained])

    def test_jacobian(self):
        x0 = np.arange(self.formulation.dimension, dtype=float)
        dx0 = x0.copy() + 1.0
        ddx0 = dx0.copy() + 1.0

        def u(x):
            ur, dur, ddur = self.formulation.recover(x, dx0, ddx0, 5.0)
            return ur

        jac_actual = self.formulation.jac_du_dx(x0, 0.0).todense()
        jac_desired = jacobian_finite_difference(u, self.formulation.no_of_dofs_unconstrained, x0)

        assert_allclose(jac_actual, jac_desired)

    def test_M(self):
        x = np.arange(self.formulation.dimension, dtype=float)
        dx = x.copy()

        M_actual = self.formulation.M(x, dx, 0.0)
        # Test last line
        assert_array_equal(M_actual.todense()[-1, :], self.B_holo_func(x[:self.no_of_dofs_unconstrained],
                                                                       0.0).todense())

        # Further test desired that tests the projected lines

    def test_F(self):
        x = np.arange(self.formulation.dimension,
                      dtype=float) + 1.0
        dx = x.copy() + 1.0
        t = 0.0
        F_actual = self.formulation.f_int(x, dx, t) - self.formulation.f_ext(x, dx, t)

        assert_array_equal(F_actual[-1], np.array([0], ndmin=1, dtype=float))

        # Further test desired that tests the projected part

    def test_D(self):
        x = np.arange(self.formulation.dimension, dtype=float)
        dx = x.copy()

        D_actual = self.formulation.D(x, dx, 0.0)
        # Test last line
        assert_array_equal(D_actual.todense()[-1, :], np.zeros((1, self.no_of_dofs_unconstrained), dtype=float))

    def test_K(self):
        x = np.arange(self.formulation.dimension, dtype=float)
        dx = x.copy()

        K_actual = self.formulation.K(x, dx, 0.0)
        # TODO: This is no real test it tests the function call

    def test_L_without_constraints(self):
        def B(u, t):
            return csr_matrix((0, 3))

        def g(u, t):
            return np.array([], dtype=float, ndmin=1)

        formulation = NullspaceConstraintFormulation(self.no_of_dofs_unconstrained, self.M_func,
                                                     self.h_func, B, self.p_func, self.h_q_func,
                                                     self.h_dq_func, g_func=g)

        x = np.arange(formulation.dimension, dtype=float)
        L_desired = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        L_actual = formulation.L(x, 0.0)
        assert_array_equal(L_actual.todense(), L_desired)
