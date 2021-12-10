# Author: Zheng Xiang
# -*- coding: utf-8 -*-
"""
Basic functions for robotics.
Reference: https://github.com/NxRLab/ModernRobotics/blob/master/packages/Python/modern_robotics/core.py
"""
import numpy as np


"""
*********************************
Representations of configuration.
*********************************
"""
def grubler_f(m, n, j, *f_i):
    """
    Formular (2.4). Grubler's formula for calculation of Degrees of Freedom (DoF) of a mechanism.
    :param m: DoF of a link. For planar mechanisms, m = 3; for spatial mechanisms, m = 6.
    :param n: Number of links.
    :param j: Number of joints.
    :param f_i: DoF brought by joint i. Should be a series of numbers, corresponds to the number of joints.
    :return: DoF of the whole mechanisms.
    """
    dof = m * (n - 1 - j) + sum(f_i)

    return dof


def vec_2_so3(x):
    """
    Formular (3.30). Calculate the skew symmetric matrix of a vector x. The group of all 3x3 skew symmetric matrix is
    so(3).
    :param x: Vector of size 3, represents the angular velocity.
    :return: 3x3 skew symmetric matrix of vector x.
    """
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def so3_2_cso3(omg, theta):
    """
    Formular (3.51). Rodrigues's formula. Convert so(3) expression into SO(3).
    :param omg: A vector represents the rotation axis. It will be scaled at first if it's not a unit vector.
    :param theta: Rotation angle.
    :return: A 3x3 rotation matrix.
    """
    # Check whether the input omg is an unit vector, if not, scale it to an unit vector
    if np.linalg.norm(omg) < 1e-6:
        pass
    elif abs(np.linalg.norm(omg) - 1) > 1e-3:
        omg = omg / np.linalg.norm(omg)
    # Calculate the rotation matrix
    skew_omg = vec_2_so3(omg)
    return np.identity(3) + np.sin(theta) * skew_omg + (1 - np.cos(theta)) * np.dot(skew_omg, skew_omg)


def cso3_2_so3(r):
    """
    Formular (3.58)-(3.61). Calculate the matrix logarithm of a rotation matrix R (Convert SO(3) expression to so(3)).
    :param r: A 3x3 rotation matrix.
    :returns: Matrix logarithm of matrix R: A skew matrix.
    """
    pass


def inv_trans(t):
    """
    Formular (3.64). Invert a homogeneous transformation matrix.
    :param t: A homogeneous transformation matrix.
    :return: The inverse of t.
    """
    r = t[0:3, 0:3]
    r_t = np.transpose(r)
    p = t[0:3, 3]
    inv_t = np.r_[np.c_[r_t, -np.dot(r_t, p)], [[0, 0, 0, 1]]]

    return inv_t


def adjoint(t):
    """
    Definition 3.20. Calculate the adjoint representation of matrix T.
    :param t: Matrix T.
    :return: An 6x6 adjoint representation of T.
    """
    r = t[0:3, 0:3]
    p = t[0:3, 3]
    ad_t = np.r_[np.c_[r, np.zeros((3, 3))], np.c_[vec_2_so3(p) * r, r]]

    return ad_t


def twist_2_screw_axis(v):
    """
    Definition (3.24). Convert a "twist" representation to a "screw axis and rotational speed" representation.

    Parameters
    ----------
    v : numpy.array
        Twist V in form of [omg, v].

    Returns
    -------
    s : numpy.array
        Screw axis.
    theta_dot : float
        Rotational speed or translational speed (If no rotation).
    """
    v = np.array(v)
    omg_v = v[0: 3]   # Omega part of twist V
    v_v = v[3:]   # v part of twist V
    if np.linalg.norm(omg_v) < 1e-6:   # No rotation
        theta_dot = np.linalg.norm(v_v)   # In this case theta_dot represents translational speed.
    else:
        theta_dot = np.linalg.norm(omg_v)
    s = v / theta_dot
    return s, theta_dot


def se3_2_cse3(s, theta):
    """
    Formular (3.88). Convert the representation of movement from "screw axis + rotation speed" to homogenous
    transformation matrix (se3 to SE3).

    Parameters
    ----------
    s : numpy.array
        Screw axis.
    theta : float
        Rotational angle.

    Returns
    -------
    np.array
        A homogenous transformation matrix.
    """
    s = np.array(s)
    omg = s[0: 3]
    v = s[3:]
    # Calculate matrix exponential
    skew_omg = vec_2_so3(omg)
    t_rt = np.matmul(np.eye(3) * theta + (1 - np.cos(theta)) * skew_omg +
                     (theta - np.sin(theta)) * np.matmul(skew_omg, skew_omg), v)   # Calculate matrix right top part
    return np.r_[np.c_[so3_2_cso3(omg, theta), t_rt], [[0, 0, 0, 1]]]


def space_jacobian(s_list, theta_list):
    """
    Formular (5.11). Calculate the space jacobian of a mechanism.

    Parameters
    ----------
    s_list : np.array
        List of screw axis of each joint as columns, relative to space frame.
    theta_list : np.array
        List of rotation angle of each joint.

    Returns
    -------
    np.array
        Space jacobian.
    """
    s_list = np.array(s_list)   # List of screw axis, as columns
    theta_list = np.array(theta_list)   # List of theta
    n = s_list.shape[1]   # Number of joints

    jacobian = s_list.copy().astype(np.float)   # Initialization of jacobian matrix
    tm = np.eye(4)
    for i in range(1, n):
        tm = np.matmul(tm, se3_2_cse3(s_list[:, i], theta_list[i]))
        jacobian[:, i] = np.matmul(adjoint(tm), np.array(s_list)[:, i])
    return jacobian


def body_jacobian(b_list, theta_list):
    """
    Formular (5.18). Calculate the body jacobian of a mechanism.

    Parameters
    ----------
    b_list : np.array
        List of screw axis of each joint as columns, relative to body frame.
    theta_list : np.array
        List of rotation angle of each joint.

    Returns
    -------
    np.array
        Body jacobian.
    """
    b_list = np.array(b_list)  # List of screw axis, as columns
    theta_list = np.array(theta_list)  # List of theta
    n = s_list.shape[1]  # Number of joints

    jacobian = b_list.copy().astype(np.float)  # Initialization of jacobian matrix
    tm = np.eye(4)
    for i in reversed(range(1, n)):
        tm = np.matmul(tm, se3_2_cse3(s_list[:, i], -theta_list[i]))
        jacobian[:, i] = np.matmul(adjoint(tm), np.array(s_list)[:, i])
    return jacobian


if __name__ == "__main__":
    t = np.array([[0, -1, 0, 1], [0, 0, -1, 2], [1, 0, 0, 3], [0, 0, 0, 1]])
    print(t)
    print(inv_trans(t))
    print(np.linalg.inv(t))
    print(adjoint(t))
    print(so3_2_cso3((0, 0.866, 0.5), 0.524))
    print(se3_2_cse3((1, 0, 0, 1, 1, 1), 0.5))
    print(abs(np.linalg.inv(se3_2_cse3((1, 0, 0, 1, 1, 1), 0.5)) - se3_2_cse3((1, 0, 0, 1, 1, 1), -0.5)) < 1e-6)
    print(twist_2_screw_axis([1, 0, 2, 1, 1, 1]))
    print(se3_2_cse3(*twist_2_screw_axis([1, 2, 3, 4, 5, 6])))
    print('*'*20)
    s_list = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, -1, 0], [0, 0, 1, 0, -2, 0], [0, 0, 0, 0, 0, 1]]).T
    theta_list = [0, 0, 0, 0]
    print(space_jacobian(s_list, theta_list))

