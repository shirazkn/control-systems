"""
These problems are taken from AAE 668 Assignment 1

Problem Descriptions
common_lyapunov :
Finding a common Lyapunov matrix for a switched system

common_lyapunov_conditional :
Stabilizing a switched system consisting two unstable subsystems
by designing the switching control as a function of state
using generalized S-procedure

sos_lyapunov_feasibility :
Finding a sum-of-squares lyapunov function for given non-linear system
by reformulating the SOS feasibility problem as an SDP problem

piecewise_quadratic_lyapunov :
Showing asymptotic stability of a switching linear system, by finding a
a piecewise-quadratic Lyapunov which is continuous at switching surface
"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


# This one works with gamma = 1,
# can be tested using the checker function
def common_lyapunov_conditional(gamma=1):

    # ones = np.array([1, 1])
    ident = np.identity(2)

    P_tilde = cp.Variable((2, 2))
    S1_tilde = cp.Variable((2, 2))
    S2_tilde = cp.Variable((2, 2))

    P = P_tilde + P_tilde.T
    S1 = S1_tilde + S1_tilde.T
    S2 = S2_tilde + S2_tilde.T

    A1 = np.array([[-1, 3], [3, -1]])
    A2 = np.array([[-1, -5], [-5, -1]])

    constraints = [
        P - ident >> 0,  # We want PD rather than PSD
        -P @ A1 - A1.T @ P - gamma*(S1 - S2) >> 0,
        -P @ A2 - A2.T @ P - gamma*(S2 - S1) >> 0
    ]

    objective = cp.Minimize(0.0)  # Feasibility Problem
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return {"P": P.value, "S1": S1.value, "S2": S2.value, "A1": A1, "A2": A2, "Problem": prob}


# Checks the results of the previous function
# over random values of x.
def common_lyapunov_conditional_checker(results):
    P, S1, S2, A1, A2 = list(results.values())[:5]

    for i in range(10):
        x = np.array([np.random.normal(scale=50), np.random.normal(scale=50)])

        if x.T @ (S1 - S2) @ x >= 0:
            if x.T @ (P @ A1 + A1.T @ P) @ x <= 0:
                continue
            else:
                print("Violated inequality 1!")
                assert False

        elif x.T @ (S1 - S2) @ x < 0:
            if x.T @ (P @ A2 + A2.T @ P) @ x <= 0:
                continue
            else:
                print("Violated inequality 2!")
                assert False
    print("All OK!")
    return


def common_lyapunov_conditional_trajectories(results, resolution=10, delta=0.005, size=100.0):
    P, S1, S2, A1, A2 = list(results.values())[:5]

    grid_values = np.linspace(-size/2, size/2, resolution)
    initial_values = []
    for value in grid_values:
        initial_values.append(np.array([value, +1 * size/2]))
        initial_values.append(np.array([value, -1 * size/2]))
        initial_values.append(np.array([+1 * size/2, value]))
        initial_values.append(np.array([-1 * size/2, value]))

    trajectories = []
    for x_0 in initial_values:
        trajectories.append({"x": [], "y": []})
        x = x_0

        # Euler iterations in time, to generate trajectory
        for t in range(10000):
            trajectories[-1]["x"].append(x[0])
            trajectories[-1]["y"].append(x[1])

            if x.T @ (S1 - S2) @ x >= 0 :
                x_dot = A1 @ x
            else:
                x_dot = A2 @ x

            x = x + x_dot * delta

    plt.figure()
    for trajectory in trajectories:
        line = plt.plot(trajectory['x'], trajectory['y'])[0]
        add_arrow(line)

    plt.xlim([-size/2, size/2])
    plt.ylim([-size / 2, size / 2])
    plt.show()
    return


def sos_lyapunov_feasibility(gamma=0.0):
    # gamma is the tolerance of positive definiteness of constraint,
    # i.e., gamma=0.0 implies positive semi-definite constraint (insufficient)
    Q_tilde = cp.Variable((2, 2))
    Q = Q_tilde + Q_tilde.T
    c11 = Q[0][0]
    c12 = Q[0][1]
    c22 = Q[1][1]

    R_tilde = cp.Variable((4, 4))
    R = R_tilde + R_tilde.T
    identity2 = np.identity(2)
    identity4 = np.identity(4)

    constraints = [
        Q - identity2 >> 0,
        R - gamma*identity4 >> 0,
        R[0][0] == 2*c11 - 2*c12,
        R[0][1] == 2*c12 - c22,
        R[0][2] == -2*c11 + c22,
        R[0][3] == c12,
        R[1][1] == 2*c22,
        R[1][2] == -2*c12,
        R[1][3] == -2*c11 + c22,
        R[2][2] == 2*c22,
        R[2][3] == c12,
        R[3][3] == 0
    ]

    objective = cp.Minimize(0.0)  # Feasibility Problem
    prob = cp.Problem(objective, constraints)
    prob.solve()
    Q = Q.value
    LfV = R.value  # Matrix corresponding to Lie derivative
    return {"Q": Q, "LfV": LfV, "Problem": prob}


def common_lyapunov():
    Q_tilde = cp.Variable((3, 3))
    Y = cp.Variable((1, 3))

    Q = Q_tilde + Q_tilde.T
    A_1 = np.array([[1, 0, 0.5], [1, 0, 0], [0, 1, 0]])
    A_2 = np.array([[-1, 1, -2], [1, 0, 0], [0, 1, 0]])
    A_3 = np.array([[1, -1, -1], [1, 0, 0], [0, 1, 0]])
    B = np.array([[1], [0], [0]])
    ident = np.identity(3)

    constraints = [
        Q - ident >> 0,
        A_1 @ Q + B @ Y + Q @ A_1.T + Y.T @ B.T >> 0,
        A_2 @ Q + B @ Y + Q @ A_2.T + Y.T @ B.T >> 0,
        A_3 @ Q + B @ Y + Q @ A_3.T + Y.T @ B.T >> 0
    ]

    objective = cp.Minimize(0.0)  # Feasibility Problem
    prob = cp.Problem(objective, constraints)
    prob.solve()
    P = np.linalg.inv(Q.value)
    K = Y @ P
    return {"P": P, "K": K.value, "Problem": prob}


def piecewise_quadratic_lyapunov(gamma=1.0):
    # Based on `Johansson & Rantzer, Computation of Piecewise Quadratic Lyapunov Functions for Hybrid Systems, 1998`
    # gamma is the tolerance of positive definiteness of constraints,
    # i.e., gamma=0.0 implies positive semi-definite constraints (insufficient)

    M_tilde = cp.Variable((2, 2))
    M = M_tilde + M_tilde.T

    U_tilde = []
    U = []
    for i in range(4):
        U_tilde.append(cp.Variable((2, 2)))
        U.append(U_tilde[-1] + U_tilde[-1].T)

    W_tilde = []
    W = []
    for i in range(4):
        W_tilde.append(cp.Variable((2, 2)))
        W.append(W_tilde[-1] + W_tilde[-1].T)

    A1 = np.array([[-1, 10], [-100, -1]])
    A2 = np.array([[-1, 100], [-10, -1]])
    A = [A1, A2, A1, A2]  # Auxiliary modes added to ensure domain of each mode is a convex polytope

    # Switching surfaces
    E = [
        np.array([[0, 1], [1, 0]]),
        np.array([[0, 1], [-1, 0]]),
        np.array([[0, -1], [-1, 0]]),
        np.array([[0, -1], [1, 0]])
    ]
    F = E

    P = []
    for i in range(4):
        P.append(F[i].T @ M @ F[i])

    ident = np.identity(2)

    constraints = []
    for i in range(4):
        constraints.append(P[i] - gamma*ident >> 0)
        constraints.append(P[i] - E[i].T @ W[i] @ E[i] - gamma*ident >> 0)
        constraints.append(-1 * (A[i].T @ P[i] + P[i] @ A[i] + E[i].T @ U[i] @ E[i]) - gamma*0.1*ident >> 0)

        # Non-negativity of U and W
        for j in range(2):
            for k in range(2):
                constraints.append(U[i][j][k] >= 0)
                constraints.append(W[i][j][k] >= 0)

    objective = cp.Minimize(0.0)
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return {"M": M.value, "P": [_P.value for _P in P], "U": [_U.value for _U in U],
            "W": [_W.value for _W in W], "E": E, "Problem": prob}


def piecewise_quadratic_lyapunov_trajectories(results, resolution=100, delta=0.005, size=100.0, contour_levels=5, initial_values=[]):
    # Plots trajectories of previous problem
    M, P, U, W, E = list(results.values())[:5]
    A1 = np.array([[-1, 10], [-100, -1]])
    A2 = np.array([[-1, 100], [-10, -1]])

    plt.figure()

    # CONTOURS
    # Adding an epsilon to the grid to avoid having points on the axes!
    x_values = np.linspace(-size/2 + np.pi/20.0, size/2, resolution)
    y_values = x_values.copy()
    z_values = np.ndarray((len(x_values), len(y_values)))

    # To plot quadratic segments separately
    # z_values1 = np.ndarray((len(x_values), len(y_values)))
    # z_values2 = np.ndarray((len(x_values), len(y_values)))

    for i in range(0, len(x_values)):
        for j in range(0, len(y_values)):
            x = np.array([[x_values[i]], [y_values[j]]])
            # z_values1[i][j] = x.T @ P[0] @ x
            # z_values2[i][j] = x.T @ P[1] @ x
            for k in range(4):
                if (E[k] @ x >= 0.0).all():
                    z_values[i][j] = x.T @ P[k] @ x
                    break

    contours = plt.contour(x_values, y_values, z_values, levels=contour_levels, cmap='Greys')
    # contours1 = plt.contour(x_values, y_values, z_values1, levels=contour_levels, alpha=alpha,
    # linestyles='dashed', cmap='Reds')
    # contours2 = plt.contour(x_values, y_values, z_values2, levels=contour_levels, alpha=alpha,
    # linestyles='dashed', cmap='Blues')

    # TRAJECTORIES
    initial_values = [np.array(i) for i in initial_values]
    trajectories = []
    for x_0 in initial_values:
        trajectories.append({"x": [], "y": []})
        x = x_0

        # Euler iterations in time, to generate trajectory
        for t in range(50000):
            trajectories[-1]["x"].append(x[0])
            trajectories[-1]["y"].append(x[1])

            if x[0]*x[1] >= 0:
                x_dot = A1 @ x
            else:
                x_dot = A2 @ x

            x = x + x_dot * delta

    for trajectory in trajectories:
        line = plt.plot(trajectory['x'], trajectory['y'])[0]
        add_arrow(line)

    plt.xlim([-size/2, size/2])
    plt.ylim([-size / 2, size / 2])
    plt.show()
    return


# Function borrowed from https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot-with-matplotlib
# Just adds an arrow indicating direction of line
def add_arrow(line, position=None, direction='right', size=15, color=None):
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()

    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )


# Prints the 'results' dict for each problem
def present(results):
    status = results["Problem"].status
    print(f"The optimization problem was solved with status : '{status}'.")
    for name in results.keys():
        if not name == "Problem":
            print(f"{name} = {results[name]}")

    return

