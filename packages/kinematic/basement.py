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
    Grubler's formula for calculation of Degrees of Freedom (DoF).
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
    Calculate the skew symmetric matrix of a vector x. The group of all 3x3 skew symmetric matrix is so(3).
    :param x: Vector of size 3, represents the angular velocity.
    :return: 3x3 skew symmetric matrix of vector x.
    """
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def so3_2_SO3(omg, theta):
    """
    Rodrigues's formula. Convert so(3) expression into SO(3).
    :param omg: A unit vector represents the rotation axis.
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
    return np.identity(3) + np.sin(theta) * vec_2_so3(omg) + (1 - np.cos(theta)) * np.dot(skew_omg, skew_omg)


def SO3_2_so3(r):
    """
    Calculate the matrix logarithm of a rotation matrix R (Convert SO(3) expression to so(3)).
    :param r: A 3x3 rotation matrix.
    :returns: Matrix logarithm of matrix R: A skew matrix.
    """
    pass


def inv_trans(t):
    """
    Invert a homogeneous transformation matrix.
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
    Calculate the adjoint representation of matrix T.
    :param t: Matrix T.
    :return: An 6x6 adjoint representation of T.
    """
    r = t[0:3, 0:3]
    p = t[0:3, 3]
    ad_t = np.r_[np.c_[r, np.zeros((3, 3))], np.c_[vec_2_so3(p) * r, r]]

    return ad_t


def twist_2_screw(v):
    v = np.array(v)
    omg_v = v[0: 3]   # Omega part of twist V
    v_v = v[3: 6]   # v part of twist V
    if np.linalg.norm(omg_v) < 1e-6:   # No rotation
        theta_dot = np.linalg.norm(v_v)
    else:
        theta_dot = np.linalg.norm(omg_v)
    s = v / theta_dot
    return s, theta_dot


def se3_2_SE3(s, theta_dot):
    """
    Convert the representation of movement from "screw axis + rotation speed" to homogenous transformation matrix (se3
    to SE3).
    :param s: Screw axis.
    :param theta_dot: Rotation speed.
    :return: A homogenous transformation matrix.
    """
    s = np.array(s)
    omg = s[0: 3]
    v = s[3: 6]
    # Calculate matrix exponential
    skew_omg = vec_2_so3(omg)
    t_rt = np.dot(np.eye(3) * theta_dot + (1 - np.cos(theta_dot) * vec_2_so3(omg)) +
                  (theta_dot - np.sin(theta_dot)) * np.dot(skew_omg, skew_omg), v)   # Calculate matrix right top part
    return np.r_[np.c_[so3_2_SO3(omg, theta_dot), t_rt], [[0, 0, 0, 1]]]


def space_jacobian(s_list, theta_list):
    s_list = np.array(s_list)
    theta_list = np.array(theta_list)
    n = s_list.shape[1]   # Number of joints
    # jacobian = np.zeros((6, n))   # Initialization of jacobian matrix
    #
    # ## Start calculation of jacobian matrix
    # jacobian[:, 0] = s_list[:, 0]   # The first column always equals S_1
    # # Calculate the list of homogenous transformation matrix
    # t_list = np.zeros((n-1, 4, 4))
    # for i in range(n-1):   # From T_1 to T_(n-1)
    #     t_list[i] = se3_2_SE3(s_list[:, i], theta_list[i])
    # # Calculate the remaining columns of jacobian matrix
    # for j in range(1, n):   # From 2 to n column of jacobian matrix
    #     t_i_minus_1 = np.eye(4)
    #     for i in range(j):   # Calculate multiplication from T_1 to T_(i-1)
    #         t_i_minus_1 = np.dot(t_i_minus_1, t_list[i])
    #     jacobian[:, i] = np.dot(adjoint(t_i_minus_1), s_list[:, j])
    jacobian = np.array(s_list).copy().astype(np.float)
    T = np.eye(4)
    for i in range(1, n):
        T = np.dot(T, se3_2_SE3(s_list[:, i], theta_list[i]))
        jacobian[:, i] = np.dot(adjoint(T), np.array(s_list)[:, i])
    return jacobian


if __name__ == "__main__":
    t = np.array([[0, -1, 0, 1], [0, 0, -1, 2], [1, 0, 0, 3], [0, 0, 0, 1]])
    print(t)
    print(inv_trans(t))
    print(np.linalg.inv(t))
    print(adjoint(t))
    print(so3_2_SO3((0, 0.866, 0.5), 0.524))
    print(se3_2_SE3((1, 0, 0, 1, 1, 1), 0.5))
    print(twist_2_screw([1, 0, 2, 1, 1, 1]))
    print(se3_2_SE3(*twist_2_screw([1, 2, 3, 4, 5, 6])))
    print('*'*20)
    s_list = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, -1, 0], [0, 0, 1, 0, -2, 0], [0, 0, 0, 0, 0, 1]]).T
    theta_list = [0, 0, 0, 0]
    print(space_jacobian(s_list, theta_list))

    # [[0.  0.  0.  0.]
    #  [0.  0.  0.  0.]
    #  [1.  1.  0.  0.]
    #  [0.  0.  0.  0.]
    #  [-1. -2. 0.  0.]
    #  [0.  0.  1.  0.]]
