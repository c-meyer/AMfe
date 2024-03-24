import numpy as np

from amfe.element import LinearSpring3D


class DummySpringMaterial:
    def __init__(self, stiffness, mass1, mass2):
        self.stiffness = stiffness
        self.mass1 = mass1
        self.mass2 = mass2


def jacobian(func, X, u, t):
    '''
    Compute the jacobian of func with respect to u using a finite differences scheme.

    '''
    ndof = X.shape[0]
    jac = np.zeros((ndof, ndof))
    h = np.sqrt(np.finfo(float).eps)
    f = func(X, u, t).copy()
    for i in range(ndof):
        u_tmp = u.copy()
        u_tmp[i] += h
        f_tmp = func(X, u_tmp, t)
        jac[:, i] = (f_tmp - f) / h
    return jac


def test_linear_spring_should_return_hook_force_in_initial_configuration():
    k = 3.0
    m1 = 1.0
    m2 = 2.0
    delta_l = 2.0
    m = DummySpringMaterial(k, m1, m2)
    spring = LinearSpring3D(m)
    X = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    u = np.array([0.0, 0.0, 0.0, delta_l, 0.0, 0.0])
    t = 0.0
    f_int_actual = spring.f_int(X, u, t)
    f_int_desired = k * delta_l * np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert np.allclose(f_int_actual, f_int_desired)


def test_linear_spring_should_return_zero_force_for_rigid_body_motion():
    tr = np.random.rand(3)  # translation
    direction = np.random.rand(3)
    direction = direction / np.linalg.norm(direction)
    k = 3.0
    m1 = 1.0
    m2 = 2.0
    m = DummySpringMaterial(k, m1, m2)
    spring = LinearSpring3D(m)
    X = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    u = np.array([tr[0], tr[1], tr[2], tr[0]-1.0+direction[0], tr[1]+direction[1], tr[2]+direction[2]])
    t = 0.0
    f_int_actual = spring.f_int(X, u, t)
    f_int_desired = k * np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(f_int_actual, f_int_desired)


def test_linear_spring_should_return_hooke_force_in_direction_of_displacement():
    n = 5
    phis = np.random.rand(n)*2*np.pi
    thetas = np.random.rand(n)*np.pi

    l0 = 1.0
    delta_l = 1.0
    l = l0 + delta_l
    k = 3.0
    m1 = 1.0
    m2 = 2.0
    m = DummySpringMaterial(k, m1, m2)
    spring = LinearSpring3D(m)
    X = np.array([0.0, 0.0, 0.0, l0, 0.0, 0.0])

    for phi, theta in zip(phis, thetas):
        x = l*np.sin(theta) * np.cos(phi)
        y = l*np.sin(theta) * np.sin(phi)
        z = l*np.cos(theta)
        u = np.array([0.0, 0.0, 0.0, x, y, z]) - X
        t = 0.0
        f_int_actual = spring.f_int(X, u, t)
        fx = np.sin(theta)*np.cos(phi) * k * delta_l
        fy = np.sin(theta)*np.sin(phi) * k * delta_l
        fz = np.cos(theta) * k * delta_l
        f_int_desired = np.array([-fx, -fy, -fz, fx, fy, fz])
        assert np.allclose(f_int_actual, f_int_desired)


def test_linear_spring_should_return_finite_difference_stiffness_matrix():
    rtol = 2E-4
    atol = 1E-6
    n = 5
    phis = np.random.rand(n)*2*np.pi
    thetas = np.random.rand(n)*np.pi
    l0 = 1.0
    delta_l = 2.0
    l = l0 + delta_l
    k = 3.0
    m1 = 1.0
    m2 = 2.0
    m = DummySpringMaterial(k, m1, m2)
    spring = LinearSpring3D(m)
    X = np.array([0.0, 0.0, 0.0, l0, 0.0, 0.0])

    for phi, theta in zip(phis, thetas):
        x = l*np.sin(theta) * np.cos(phi)
        y = l*np.sin(theta) * np.sin(phi)
        z = l*np.cos(theta)
        u = np.array([0.0, 0.0, 0.0, x, y, z]) - X
        t = 0.0
        K, f = spring.k_and_f_int(X, u, t)
        K_finite_diff = jacobian(spring.f_int, X, u, t=0)
        np.testing.assert_allclose(K, K_finite_diff, rtol=rtol, atol=atol)


def test_linear_spring_should_return_hook_stiffness_matrix_for_linearization_around_initial_configuration():
    k = 3.0
    m1 = 1.0
    m2 = 2.0
    m = DummySpringMaterial(k, m1, m2)
    spring = LinearSpring3D(m)
    X = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    u = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    t = 0.0
    k_actual = spring.k_int(X, u, t)
    k_desired = k * np.array([
        [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -0.0, 0.0, 0.0]
    ])
    assert np.allclose(k_actual, k_desired)


def test_linear_spring_should_return_hook_stiffness_matrix_for_linearization_around_90_degrees_rotated_configuration():
    k = 3.0
    m1 = 1.0
    m2 = 2.0
    m = DummySpringMaterial(k, m1, m2)
    spring = LinearSpring3D(m)
    X = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    u = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 1.0])
    t = 0.0
    k_actual = spring.k_int(X, u, t)
    k_desired = k * np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, 0.0, 1.0]
    ])
    assert np.allclose(k_actual, k_desired)
