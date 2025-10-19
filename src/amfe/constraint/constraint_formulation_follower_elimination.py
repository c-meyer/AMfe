#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

import numpy as np
from scipy.sparse import csr_matrix, issparse, coo_matrix
from scipy.sparse.linalg import spsolve

from .constraint_formulation import ConstraintFormulationBase
from ..logging import log_debug, log_warning


class FollowerEliminationConstraintFormulation(ConstraintFormulationBase):
    """
    Works only with holonomic scleronomic constraints that result in a constant B matrix
    (Jacobian of the constraint function)

    Attributes
    ----------
    _L: csr_matrix
        Matrix that is able to eliminate the constrained dofs by applying :math:`L^T A L` to a matrices A
    _L_changed: bool
        Internal flag that indicates if L must be updated when it is asked for the next time

    Notes
    -----
    Currently there is no check if this formulation is allowed to use!
    """
    def __init__(self, no_of_dofs_unconstrained, M_func, h_func, B_func, p_func=None,
                 jac_h_u=None, jac_h_du=None, jac_p_u=None, jac_p_du=None,
                 g_func=None, b_func=None, a_func=None):
        super().__init__(no_of_dofs_unconstrained, M_func, h_func, B_func, p_func,
                         jac_h_u, jac_h_du, jac_p_u, jac_p_du,
                         g_func, b_func, a_func)
        self._L_cached = None
        self._L_changed = True  # Setting flag for lazy evaluation

    @property
    def dimension(self):
        """
        Returns the dimension of the system after constraints have been applied

        Returns
        -------
        dim: int
            dimension of the system after constraints are applied
        """
        return self._L.shape[1]

    @property
    def _L(self):
        """
        Returns the L matrix that is able to eliminate the constrained dofs by applying :math:`L^T A L` to a matrices A

        Returns
        -------
        L: csr_matrix
            The matrix L
        """
        if self._L_changed:
            log_debug(__name__, "Lazy load: L Follower Elimination matrix must be recomputed.")
            self._compute_L()
            self._L_changed = False
        return self._L_cached

    def update(self):
        """
        Function that is called by observers if state has changed

        Returns
        -------
        None
        """
        # This class assumes that the C matrix is constant and Boolean
        self._L_changed = True

    def _compute_L(self):
        """
        Internal function that computes the matrix L

        The function is called when L must be updated
        L is the nullspace of B

        Returns
        -------
        None
        """
        # Follower elimination assumes that B is constant (linear, scleronomic) and independent on q!
        # Thus, B is called by just calling for any arbitrary values, q and t
        q = np.zeros(self._no_of_dofs_unconstrained, dtype=np.float64)
        t = 0.0
        B = self._B_func(q, t)
        constrained_dofs = self._get_constrained_dofs_by_B(B)
        if issparse(B):
            self._L_cached = self._get_L_by_constrained_dofs(B, constrained_dofs, B.shape[1], format='csr')
        else:
            self._L_cached = self._get_L_by_constrained_dofs(B, constrained_dofs, B.shape[1], format='dense')

    @staticmethod
    def _get_constrained_dofs_by_B(B):
        """
        Static method that computes the indices of those dofs that are constrained when a matrix B is given

        Parameters
        ----------
        B: csr_matrix
            B is a matrix coming from the constraint definitions: B q + b = 0

        Returns
        -------

        """
        constrained_dofs = []
        # check if only one 1 is in each row:

        # TODO: The following identification of follower dofs is a simple heuristic that might need improvement.
        nrows, ncols = B.shape
        for row in range(nrows):
            found = False
            b_row = B[row, :]
            if issparse(B):
                b_row = b_row.toarray().flatten()
            sorted_dofs = np.argsort(np.abs(b_row))[::-1]
            for dof in sorted_dofs:
                # debugging message:
                if b_row[dof] != 1.0 and b_row[dof] != -1.0:
                    log_debug(__name__, 'Found dof for elimination might not be the best one.')
                if dof not in constrained_dofs:
                    constrained_dofs.append(dof)
                    found = True
                    if b_row[dof] == 0.0:
                        raise NotImplementedError('This constraint cannot be handled with AMfe yet because the'
                                                  'identified dof for elimination is a zero in the B matrix.')
                    ratio = np.max(np.abs(b_row)) / b_row[dof]
                    if np.max(np.abs(b_row))/b_row[dof] > 1.e4:
                        log_warning(__name__, 'The ratio {} for this constraint is large. The constraint might lead to a ill-conditioned system.'.format(ratio))
                    break
            if not found:
                raise ValueError("Could not find any dof that can be eliminated in B")
        constrained_dofs_set = set(constrained_dofs)
        assert len(constrained_dofs_set) == len(constrained_dofs)
        return constrained_dofs

    @staticmethod
    def _get_L_by_constrained_dofs(B, constrained_dofs, ndof_unconstrained, format='csr'):
        """
        Internal static function that computes L by given indices of constrained dofs

        Parameters
        ----------
        constrained_dofs: list or ndarray
            list containing the indices of the constrained dofs
        ndof_unconstrained: int
            number of dofs of the unconstrained system
        format: str
            format = 'csr' or 'dense' describes the format of L

        Returns
        -------
        L: csr_matrix
            computed L matrix
        """
        assert B.shape[0] == len(constrained_dofs)
        assert len(set(constrained_dofs)) == len(constrained_dofs)

        i_master = np.arange(ndof_unconstrained - len(constrained_dofs))
        master_dofs = np.arange(0, ndof_unconstrained)
        master_dofs = np.delete(master_dofs, constrained_dofs)
        assert len(i_master) == len(master_dofs)

        # Unity for master dofs
        rows_master = master_dofs
        cols_master = np.arange(len(master_dofs))
        vals_master = np.ones(len(master_dofs))
        mat_master = coo_matrix((vals_master, (rows_master, cols_master)), shape=(ndof_unconstrained, len(master_dofs)))

        B_s = B[:, constrained_dofs]
        B_m = B[:, master_dofs]
        L_s = spsolve(B_s, -B_m)
        if issparse(L_s):
            L_s = L_s.tocoo()
        else:
            L_s = coo_matrix(L_s)
        rows_follower = np.array([constrained_dofs[r] for r in L_s.row])
        cols_follower = L_s.col
        data_follower = L_s.data
        mat_follower = coo_matrix((data_follower, (rows_follower, cols_follower)), shape=(ndof_unconstrained, len(master_dofs)))
        L = mat_master + mat_follower

        if format == 'csr':
            return L.tocsr()
        elif format == 'dense':
            return L.toarray()
        else:
            raise ValueError('Only csr or dense format allowed')

    def u(self, x, t):
        """

        Parameters
        ----------
        x: numpy.array
            Global state vector of the system
        t: float
            time

        Returns
        -------
        u: numpy.array
            recovered displacements of the unconstrained system

        """
        return self._L.dot(x)

    def jac_du_dx(self, x, t):
        """
        Returns the jacobian of the displacements w.r.t. the state vector.

        Parameters
        ----------
        x: numpy.array
            Global state vector of the system
        t: float
            time

        Returns
        -------
        jac_du_dx: csr_matrix
            Jacobian of the displacements w.r.t. the state vector.

        """
        return self._L

    def du(self, x, dx, t):
        """

        Parameters
        ----------
        x: numpy.array
            Global state vector of the system
        dx: numpy.array
            First time derivative of global state vector of the constrained system
        t: float
            time

        Returns
        -------
        du: numpy.array
            recovered velocities of the unconstrained system

        """
        return self._L.dot(dx)

    def ddu(self, x, dx, ddx, t):
        """

        Parameters
        ----------
        x: numpy.array
            Global state vector of the system
        dx: numpy.array
            First time derivative of global state vector of the constrained system
        ddx: numpy.array
            Second time derivative of global state vector of the constrained system
        t: float
            time

        Returns
        -------
        ddu: numpy.array
            recovered accelerations of the unconstrained system

        """
        return self._L.dot(ddx)

    def lagrange_multiplier(self, x, t):
        """
        Recovers the lagrange multipliers of the unconstrained system

        Parameters
        ----------
        x: numpy.array
            Global state vector of the system
        t: float
            time

        Returns
        -------
        lambda_: numpy.array
            recovered displacements of the unconstrained system

        """
        return np.array([], ndmin=1, dtype=np.float64)

    def M(self, x, dx, t):
        r"""
        Returns the constrained mass matrix

        Parameters
        ----------
        x: numpy.array
            Global state vector of the system
        dx: numpy.array
            First time derivative of global state vector of the constrained system
        t: float
            time

        Returns
        -------
        M: csr_matrix
            Constrained mass matrix

        Notes
        -----
        In this formulation this returns

        .. math::
            L^T M_{raw} L

        """
        u = self.u(x, t)
        du = self.du(x, dx, t)
        return self._L.T.dot(self._M_func(u, du, t)).dot(self._L)

    def f_int(self, x, dx, t):
        r"""
        Returns the constrained f_int vector

        Parameters
        ----------
        x: numpy.array
            Global state vector of the system
        dx: numpy.array
            First time derivative of global state vector of the constrained system
        t: float
            time

        Returns
        -------
        F: numpy.array
            Constrained F vector

        Notes
        -----
        In this formulation this returns

        .. math::
            L^T h(u, du, t)

        """

        u = self.u(x, t)
        du = self.du(x, dx, t)
        return self._L.T.dot(self._h_func(u, du, t))

    def f_ext(self, x, dx, t):
        r"""
        Returns the constrained f_ext vector

        Parameters
        ----------
        x: numpy.array
            Global state vector of the system
        dx: numpy.array
            First time derivative of global state vector of the constrained system
        t: float
            time

        Returns
        -------
        F: numpy.array
            Constrained F vector

        Notes
        -----
        In this formulation this returns

        .. math::
            L^T p(u, du, t)

        """

        u = self.u(x, t)
        du = self.du(x, dx, t)
        return self._L.T.dot(self._p_func(u, du, t))

    def K(self, x, dx, t):
        r"""
        Returns the constrained stiffness matrix

        Parameters
        ----------
        x: numpy.array
            Global state vector of the system
        dx: numpy.array
            First time derivative of global state vector of the constrained system
        t: float
            time

        Returns
        -------
        K: csr_matrix
            Constrained mass matrix

        Notes
        -----
        In this formulation this returns

        .. math::
            L^T \frac{\mathrm{d}(h-p)}{\mathrm{d} u} L

        """
        u = self.u(x, t)
        du = self.du(x, dx, t)
        if self._jac_h_u is not None:
            if self._jac_p_u is not None:
                return self._L.T.dot(self._jac_h_u(u, du, t) - self._jac_p_u(u, du, t)).dot(self._L)
            else:
                return self._L.T.dot(self._jac_h_u(u, du, t)).dot(self._L)
        else:
            raise NotImplementedError('Numerical differentiation of h is not implemented yet')

    def D(self, x, dx, t):
        r"""
        Returns the constrained damping matrix

        Parameters
        ----------
        x: numpy.array
            Global state vector of the system
        dx: numpy.array
            First time derivative of global state vector of the constrained system
        t: float
            time

        Returns
        -------
        D: csr_matrix
            Constrained damping matrix

        Notes
        -----
        In this formulation this returns

        .. math::
            L^T \frac{\mathrm{d}(h-p)}{\mathrm{d} \dot{u}} L

        """
        u = self.u(x, t)
        du = self.du(x, dx, t)
        if self._jac_h_du is not None:
            if self._jac_p_du is not None:
                return self._L.T.dot(self._jac_h_du(u, du, t) - self._jac_p_du(u, du, t)).dot(self._L)
            else:
                return self._L.T.dot(self._jac_h_du(u, du, t)).dot(self._L)
        else:
            raise NotImplementedError('Numerical differentiation of h is not implemented yet')
