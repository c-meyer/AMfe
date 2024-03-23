#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
3D linear Beam element
"""

__all__ = [
    'LinearSpring3D'
]

import numpy as np

from .element import Element

# try to import Fortran routines
use_fortran = False
try:
    import amfe.f90_element
    use_fortran = True
except ModuleNotFoundError:
    print('Python was not able to load the fast fortran element routines.')


class LinearSpring3D(Element):
    """
    Linear Spring Element
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.K = np.zeros((6, 6))
        self.M = np.zeros((6, 6))
        self.f = np.zeros(6)

    def fields(self):
        return ('ux', 'uy', 'uz')

    def dofs(self):
        return (('N', 0, 'ux'),
                ('N', 0, 'uy'),
                ('N', 0, 'uz'),
                ('N', 1, 'ux'),
                ('N', 1, 'uy'),
                ('N', 1, 'uz'),
                )

    def _compute_tensors(self, X, u, t):
        # X_mat = X.reshape(-1, 3)
        # l = np.linalg.norm(X_mat[1,:]-X_mat[0,:])

        # Element stiffness matrix in X direction
        # k_el_loc = np.zeros((6, 6), dtype=np.float64)
        k = self.material.stiffness

        x = X + u
        X_mat = X.reshape(-1, 3)
        x_mat = x.reshape(-1, 3)
        v_k = x_mat[1, :] - x_mat[0, :]
        l = np.linalg.norm(v_k)
        l0 = np.linalg.norm(X_mat[1, :] - X_mat[0, :])
        delta_l = l - l0
        e_k = v_k / l

        e_X = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        e_Y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        e_Z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        fx = k * delta_l * e_k.dot(e_X)
        fy = k * delta_l * e_k.dot(e_Y)
        fz = k * delta_l * e_k.dot(e_Z)

        f = np.array([-fx, -fy, -fz, fx, fy, fz], dtype=np.float64)

        lx = x_mat[1, 0] - x_mat[0, 0]  # x2 - x1
        ly = x_mat[1, 1] - x_mat[0, 1]  # y2 - y1
        lz = x_mat[1, 2] - x_mat[0, 2]  # z2 - z1

        Fx_u1 = k * (-delta_l / l + delta_l* lx*lx / l**3 - lx*lx / l**2)
        Fx_v1 = k * (delta_l * lx*ly / l**3 - lx * ly / l**2)
        Fx_w1 = k * (delta_l * lx*lz / l**3 - lx * lz / l**2)

        Fy_u1 = k * (delta_l * lx*ly / l**3 - lx * ly / l**2)
        Fy_v1 = k * (-delta_l / l + delta_l * ly*ly / l**3 - ly*ly / l**2)
        Fy_w1 = k * (delta_l * ly * lz / l**3 - ly * lz / l**2)

        Fz_u1 = k * (delta_l * lx * lz / l**3 - lx * lz / l**2)
        Fz_v1 = k * (delta_l * ly * lz / l**3 - ly * lz / l**2)
        Fz_w1 = k * (-delta_l / l + delta_l * lz*lz / l**3 - lz*lz / l**2)


        Fx_u2 = -Fx_u1
        Fx_v2 = -Fx_v1
        Fx_w2 = -Fx_w1
        Fy_u2 = -Fy_u1
        Fy_v2 = -Fy_v1
        Fy_w2 = -Fy_w1
        Fz_u2 = -Fz_u1
        Fz_v2 = -Fz_v1
        Fz_w2 = -Fz_w1

        k_el = np.array([
            [-Fx_u1, -Fx_v1, -Fx_w1, -Fx_u2, -Fx_v2, -Fx_w2],
            [-Fy_u1, -Fy_v1, -Fy_w1, -Fy_u2, -Fy_v2, -Fy_w2],
            [-Fz_u1, -Fz_v1, -Fz_w1, -Fz_u2, -Fz_v2, -Fz_w2],
            [Fx_u1, Fx_v1, Fx_w1, Fx_u2, Fx_v2, Fx_w2],
            [Fy_u1, Fy_v1, Fy_w1, Fy_u2, Fy_v2, Fy_w2],
            [Fz_u1, Fz_v1, Fz_w1, Fz_u2, Fz_v2, Fz_w2],
        ], dtype=np.float64)


        self.K = k_el
        self.M = np.zeros((6, 6))
        self.f = f
        self.S = np.zeros((2, 6))
        self.E = np.zeros((2, 6))
        return

    def _m_int(self, X, u, t):
        self._compute_tensors(X, u, t)
        return self.M
