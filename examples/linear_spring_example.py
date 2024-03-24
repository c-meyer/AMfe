"""
This example shows how one can simulate a harmonic oscillator with AMfe.

The example consists of a spring that is fixed on the lower end, and free on the upper end.
A lumped mass is located at the upper end.
The system is excited by a harmonic force.
"""

from amfe.ui import (create_structural_component, assign_material_by_elementids, set_dirichlet_by_group,
                     create_mechanical_system, solve_nonlinear_dynamic)
from amfe.mesh import Mesh
from amfe.material import LinearSpringMaterial
from amfe.element import LinearSpring3D

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Create a 3D mesh with two nodes.
    mesh = Mesh(dimension=3)
    node1 = mesh.add_node((0.0, 0.0, 0.0))
    node2 = mesh.add_node((0.0, 0.0, 1.0))

    # Define a group for each node for easier access.
    mesh.create_group("fixation", [node1])
    mesh.create_group("free", [node2])

    # Create a spring Element in the mesh.
    element = LinearSpring3D()
    shape = "straight_line"
    element1 = mesh.add_element(shape, [node1, node2])

    # Create a AMfe Structural Component from the mesh.
    component = create_structural_component(mesh)

    # Define the spring material and assign it to the spring element.
    k = 3.0
    m1 = 0.0
    m2 = 2.0
    spring = LinearSpringMaterial(k, m1, m2)
    assign_material_by_elementids(component, spring, [element1])

    # Set Dirichlet Boundary conditions.
    set_dirichlet_by_group(component, "fixation", direction=('ux', 'uy', 'uz'))
    set_dirichlet_by_group(component, "free", direction=('uy',))

    # Get the system object and the constraint formulation
    system, formulation = create_mechanical_system(component, True)

    # Now, we apply external forces by monkeypatching the f_ext method of the system object.
    # We cannot apply this Neumann condition by using AMfe's API directly because this API only allows to
    # add boundary conditions on boundary elements like lines for 2D meshes or surfaces for 3D meshes.
    # But we want to create a point force. This is not implemented in AMfe, but can be applied by
    # monkeypatching the f_ext method of the system object.

    # Define a position vector for the external force: The force is applied on the top node in z-direction
    # This is the second degree of freedom (2/2) of the system
    b = np.array([0.0, 1.0])

    def my_f_ext(u, du, t):
        amp = 4.0
        return amp*np.sin(t)*b

    system.f_ext = my_f_ext

    # Solve the system with default dynamic solver.
    sol = solve_nonlinear_dynamic(system, formulation, component, 0.0, 8*np.pi, 0.1, 1)

    # Plot the x- and z- displacement of the upper node:
    # The x-displacement is the 4-th dof (=position 3 in q vector).
    # The z-displacement is the 6-th dof (=position 5 in q vector).
    f, (ax1, ax2) = plt.subplots(2, 1)
    qx = [e[3] for e in sol.q]
    qz = [e[5] for e in sol.q]
    ax1.plot(sol.t, qx)
    ax2.plot(sol.t, qz)
    plt.show()
