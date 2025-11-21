#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
ECSW assembly.

Provides Assembly routines for the ECSW Assembly
"""

import numpy as np
from scipy.sparse import coo_matrix

from amfe.assembly.structural_assembly import StructuralAssembly
from amfe.assembly.tools import fill_csr_matrix

__all__ = [
    'EcswGAssembly'
]


class EcswGAssembly(StructuralAssembly):
    """
    Class handling assembly for G
    """

    def __init__(self):
        """

        Parameters
        ----------
        """
        super().__init__()

    def preallocate(self, no_of_dofs, elements2global):
        data = np.zeros(np.sum([len(e) for e in elements2global]))
        i = np.zeros_like(data)
        j = np.zeros_like(data)
        K_coo = coo_matrix((data, (i, j)), shape=(no_of_dofs, len(elements2global)))
        return K_coo

    def assemble_k_and_f(self, nodes, ele_objects, connectivities, elements2dofs, dofvalues=None, t=0., K_csr=None,
                         f_glob=None):
        """
        Assemble the tangential stiffness matrix and nonliner internal or external force vector.

        This method can be used for assembling K_int and f_int or for assembling K_ext and f_ext depending on which
        ele_objects and connectivities are passed

        Parameters
        ----------
        nodes : ndarray
            Node Coordinates
        ele_objects : ndarray
            Ndarray with Element objects that shall be assembled
        connectivities : list of ndarrays
            Connectivity of the elements mapping to the indices of nodes ndarray
        elements2dofs : list of ndarrays
            Mapping the elements to their global dofs
        dofvalues : ndarray
            current values of all dofs (at time t)
        t : float
            time. Default: 0.
        K_csr : csr_matrix (optional)
            A preallocated csr_matrix can be passed for faster assembly
        f_glob : numpy.array (optional)
            A preallocated numpy.array can be passede for faster assembly

        Returns
        --------
        K : csr_matrix
            global stiffness matrix
        f : ndarray
            global internal force vector

        """

        K_coo = K_csr

        if dofvalues is None:
            maxdof = np.max([np.max(e) for e in elements2dofs])
            dofvalues = np.zeros(maxdof + 1)

        if K_coo is None or not isinstance(K_coo, coo_matrix):
            no_of_dofs = np.max([np.max(e) for e in elements2dofs]) + 1
            K_coo = self.preallocate(no_of_dofs, elements2dofs)

        if f_glob is None:
            f_glob = np.zeros(K_csr.shape[0], dtype=np.float64)

        K_coo.data[:] = 0.0
        K_coo.row[:] = 0.0
        K_coo.col[:] = 0.0
        f_glob[:] = 0.0

        # loop over all elements
        # (i - element index, indices - DOF indices of the element)
        element_number = 0
        current_coordinate = 0
        for ele_obj, connectivity, globaldofindices in zip(ele_objects, connectivities, elements2dofs):
            # X - undeformed positions of the i-th element
            X_local = nodes[connectivity, :].reshape(-1)
            # displacements of the i-th element
            u_local = dofvalues[globaldofindices]
            # computation of the element tangential stiffness matrix and nonlinear force
            _, f_local = ele_obj.k_and_f_int(X_local, u_local, t)
            # adding the local force to the global one
            num_dofs = len(globaldofindices)
            K_coo.data[current_coordinate:current_coordinate+num_dofs] = f_local
            K_coo.row[current_coordinate:current_coordinate+num_dofs] = globaldofindices[:]
            K_coo.col[current_coordinate:current_coordinate+num_dofs] = element_number
            current_coordinate += num_dofs
            element_number += 1
        return K_coo, f_glob

    def assemble_k_f_S_E(self, nodes, ele_objects, connectivities, elements2dofs, elements_on_node, dofvalues=None,
                         t=0, K_csr=None, f_glob=None):
        """
        Assemble the stiffness matrix with stress recovery of the given mesh and element.

        Parameters
        ----------
        nodes_df : pandas.DataFrame
            Node Coordinates
        ele_objects : ndarray
            Ndarray with Element objects that shall be assembled
        connectivities : list of ndarrays
            Connectivity of the elements mapping to the indices of nodes ndarray
        elements2dofs : list of ndarrays
            Mapping the elements to their global dofs
        elements_on_node : ndarray
            ndarray containing number of elements that are assembled belonging to a node
        dofvalues : ndarray
            current values of all dofs (at time t) ordered by the dofnumbers given by elements2dof list
        t : float
            time. Default: 0.
        K_csr : csr_matrix (optional)
            A preallocated csr_matrix can be passed for faster assembly
        f_glob : numpy.array
            A preallocated numpy.array can be passed for faster assembly

        Returns
        --------
        K : csr_matrix
            global stiffness matrix
        f : ndarray
            global internal force vector
        S : pandas.DataFrame
            unconstrained assembled stress tensor
        E : pandas.DataFrame
            unconstrained assembled strain tensor
        """
        raise NotImplementedError('Stress and strain evaluation is not applicable.')
