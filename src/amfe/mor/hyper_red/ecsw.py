"""
TODO: Write introduction to ECSW
"""

import numpy as np
from scipy.linalg import solve as linsolve
from scipy.sparse import csc_matrix

from amfe.logging import log_debug, log_info
from .ecsw_assembly import EcswAssembly

__all__ = ['sparse_nnls',
           'ecsw_assemble_G_and_b',
           'ecsw_get_weights_by_component',
           'EcswAssembly']


def sparse_nnls(G, b, tau, conv_stats=True):
    r"""
    Run the sparse NNLS-solver in order to find a sparse vector xi satisfying

    .. math::
        || G \xi - b ||_2 \leq \tau ||b||_2 \quad\text{with}\quad \min||\xi||_0

    Parameters
    ----------
    G : ndarray, shape: (n*m, no_of_elements)
        force contribution matrix
    b : ndarray, shape: (n*m)
        force contribution vector
    tau : float
        tolerance
    conv_stats : bool
        Flag for setting, that more detailed output is produced with
        convergence information.

    Returns
    -------
    x : csc_matrix
        sparse vector containing the weights
    stats : ndarray
        Infos about the convergence of the system. The first column shows the
        size of the active set, the second column the residual. If conv_info is
        set to False, an empty array is returned.

    References
    ----------
    .. [1]  C. L. Lawson and R. J. Hanson. Solving least squares problems,
            volume 15. SIAM, 1995.

    .. [2]  T. Chapman, P. Avery, P. Collins, and C. Farhat. Accelerated mesh
            sampling for the hyper reduction of nonlinear computational models.
            International Journal for Numerical Methods in Engineering, 2016.

    """
    no_of_elements = G.shape[1]
    norm_b = np.linalg.norm(b)
    r = b

    xi = np.zeros(no_of_elements) # the resulting vector
    zeta = np.zeros(no_of_elements) # the trial vector which is iterated over

    # Boolean active set; allows quick and easys indexing through masking with
    # high performance at the same time
    active_set = np.zeros(no_of_elements, dtype=bool)

    stats = []
    while np.linalg.norm(r) > tau * norm_b:
        mu = G.T @ r
        idx = np.argmax(mu)
        if active_set[idx] == True:
            raise RuntimeError('snnls: The index has {} has already been added and is considered to be the best again.')
        active_set[idx] = True
        print('Added element {}'.format(idx))
        while True:
            # Trial vector zeta is solved for the sparse solution
            zeta[~active_set] = 0.0
            G_red = G[:, active_set]
            zeta[active_set] = linsolve(G_red.T @ G_red, G_red.T @ b)

            # check, if gathered solution is full positive
            if np.min(zeta[active_set]) >= 0.0:
                xi[:] = zeta[:]
                break
            # remove the negative elements from the active set
            # Get all elements which violate the constraint, i.e. are in the
            # active set and are smaller than zero
            mask = np.logical_and(zeta <= 0.0, active_set)

            ele_const = np.argmin(xi[mask] / (xi[mask] - zeta[mask]))
            const_idx = np.where(mask)[0][ele_const]
            print('Remove element {} '.format(const_idx) +
                   'violating the constraint.')
            # Amplify xi with the difference of zeta and xi such, that the
            # largest mismatching negative point becomes zero.
            alpha = np.min(xi[mask] / (xi[mask] - zeta[mask]))
            xi += alpha * (zeta - xi)
            # Set active set manually as otherwise floating point roundoff
            # errors are not considered.
            # active_set = xi != 0
            active_set[const_idx] = False

        r = b - G[:, active_set] @ xi[active_set]
        log_debug(__name__, "snnls: residual {} No of active elements: {}".format(np.linalg.norm(r), len(np.where(xi)[0])))
        if conv_stats:
            stats.append((len(np.where(xi)[0]), np.linalg.norm(r)))

    # sp.optimize.nnls(A, b)
    indices = np.where(xi)[0]  # remove the nasty tuple from np.where()
    xi_red = xi[active_set]
    indptr = np.array([0, len(xi_red)])
    x = csc_matrix((xi_red, indices, indptr), shape=(G.shape[1], 1))
    if conv_stats and not stats:
        stats.append((0, np.linalg.norm(r)))
    stats = np.array(stats)
    return x, stats


def ecsw_assemble_G_and_b(component, S, W, timesteps=None):
    """
    Assembles the element contribution matrix G for the given snapshots S.

    This function is needed for cubature bases Hyper reduction methods
    like the ECSW.

    Parameters
    ----------
    component : amfe.MeshComponent
        amfe.Component, if a reduction basis should be used, it should already
        be the component that is reduced by this reduction basis
    S : ndarray, shape (no_of_dofs, no_of_snapshots)
        Snapshots gathered as column vectors.
    W : ndarray
        projection matrix
    timesteps : ndarray, shape(no_of_snapshots)
        the timesteps of where the snapshots have been generated can be passed,
        this is important for systems with certain constraints

    Returns
    -------
    G : ndarray, shape (n*m, no_of_elements)
        Contribution matrix of internal forces. The columns form the
        internal force contributions on the basis V for the m snapshots
        gathered in S.
    b : ndarray, shape (n*m, )
        summed force contribution

    Note
    ----
    This assembly works on constrained variables
    """
    # Check the raw dimension
    # Currently not applicable
    # assert(component.no_of_dofs == S.shape[0])
    if timesteps is None:
        timesteps = np.zeros(S.shape[1], dtype=np.float64)

    no_of_dofs, no_of_snapshots = S.shape
    no_of_reduced_dofs = W.shape[1]

    no_of_elements = component.no_of_elements
    log_info(__name__, 'Start building large selection matrix G. In total {0:d} elements are treated:'.format(
                  no_of_elements))

    G = np.zeros((no_of_reduced_dofs*no_of_snapshots, no_of_elements))

    # Temporarily replace Assembly of component:
    old_assembly = component.assembly
    g_assembly = EcswAssembly([], [])
    component.assembly = g_assembly

    # Weight only one element by one
    g_assembly.weights = [1.0]

    # Set dq and ddq = 0
    dq = np.zeros(no_of_dofs)

    # loop over all elements
    for element_no in range(no_of_elements):
        # Change nonzero weighted elements to current element
        g_assembly.indices = [element_no]

        log_debug(__name__, 'Assemble element {:10d} / {:10d}'.format(element_no+1, no_of_elements))
        # loop over all snapshots

        for snapshot_number, (snapshot_vector, t) in enumerate(zip(S.T, timesteps)):
            G[snapshot_number*no_of_reduced_dofs:(snapshot_number+1)*no_of_reduced_dofs, element_no] = W.T @ component.f_int(snapshot_vector,
                                                                                                       dq, t)

    b = np.sum(G, axis=1)

    # reset assembly
    component.assembly = old_assembly
    return G, b


def ecsw_get_weights_by_component(component, S, W, timesteps=None, tau=0.001, conv_stats=True):
    """
    Reduce the given MeshComponent

    Parameters
    ----------
    component : instance of MeshComponent
        MeshComponent
    S : ndarray, shape (no_of_dofs, no_of_snapshots)
        Snapshots
    W : ndarray
        projection basis
    timesteps : ndarray, optional
        timesteps of the training snapshots
        if None, all timesteps will be set to zero
    tau : float
        tolerance of the ECSW reduction
    conv_stats : bool
        Flag if conv_stats shall be collected

    Returns
    -------
    weights : ndarray
        ecsw weights
    indices : ndarray
        row based indices of elements that have non-zero weights
    stats : ndarray
        convergence stats of the snnls solver
    """

    if timesteps is None:
        timesteps = np.zeros(S.shape[1], dtype=np.float64)

    # Create G and b from snapshots:
    G, b = ecsw_assemble_G_and_b(component, S, W, timesteps)

    weights, indices, stats = ecsw_get_weights_by_G_and_b(G, b, tau, conv_stats)

    return weights, indices, stats


def ecsw_get_weights_by_G_and_b(G, b, tau, conv_stats):
    # Calculate indices and weights
    x, stats = sparse_nnls(G, b, tau, conv_stats)
    indices = x.indices
    weights = x.data

    return weights, indices, stats
