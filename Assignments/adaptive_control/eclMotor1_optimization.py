import cvxpy as cp
import numpy as np

def controller_search(k1=None, k2=None, gamma=0.01, alpha=0.0):
    # gamma is the tolerance of positive definiteness of constraints,
    # Use non-zero alpha to solve for exponential stability
    # i.e., gamma=0.0 implies positive semi-definite constraints (insufficient)

    P_tilde = cp.Variable((3, 3))
    P = P_tilde + P_tilde.T

    # Optimization parameters
    k = [k1, k2]

    kf = 1000
    omega_y = 2*np.pi/0.06
    Me_min = 0.025;
    Me_max = 0.085;
    B_min = 0.1;
    B_max = 0.35;
    Acog_max = 0.05;

    A_0_min = np.array([[0, 1, 0], [k[0] / Me_max, -B_max / Me_min, k[1] / Me_max], [1, 0, 0]])
    A_0_max = np.array([[0, 1, 0], [k[0] / Me_min, -B_min / Me_max, k[1] / Me_min], [1, 0, 0]])
    delA_1 = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 0]])
    delA_2 = np.array([[0, 0, 0], [-1, 0, 0], [0, 0, 0]])
    delA_3 = np.array([[0, 0, 0], [-1, 0, 0], [0, 0, 0]])
    m1 = 0
    M1 = kf / Me_min
    M2 = omega_y * Acog_max / Me_min
    m2 = -M2
    M3 = 3 * omega_y * Acog_max / Me_min
    m3 = -M3

    A_0s = [A_0_min, A_0_max]
    m1s = [m1, M1]
    m2s = [m2, M2]
    m3s = [m3, M3]

    ident = np.identity(3)

    constraints = [P - gamma * ident >> 0]

    gamma_alpha = gamma if alpha == 0.0 else 0.0

    for i1 in range(2):
        for i2 in range(2):
            for i3 in range(2):
                for i4 in range(2):
                    Ai = A_0s[i1] + m1s[i2]*delA_1 + m2s[i3]*delA_2 + m3s[i4]*delA_3
                    constraints.append(P @ Ai + np.transpose(Ai) @ P + alpha*P + gamma_alpha*ident <= 0)

    objective = cp.Minimize(0.0)
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return {"Parameters": {"k1": k[0], "k2": k[1]}, "P": [_P.value for _P in P], "Problem": prob}


# Prints the 'results' dict for each problem
def present(results):
    status = results["Problem"].status
    print(f"The optimization problem was solved with status : '{status}'.")
    for name in results.keys():
        if not name == "Problem":
            print(f"{name} = {results[name]}")
    return


#k1s = np.linspace(-1.0, -100.0, 50)
#k2s = np.linspace(-1.0, -100.0, 50)
#solved = False
#for k1 in k1s:
#    if solved:
#        break
#    for k2 in k2s:
#        prob = controller_search(k1=k1, k2=k2, alpha=1.0)
#        if prob["Problem"].status == "optimal":
#            present(prob)
#            solved = True
#            break

#prob = controller_search(k1=-500, k2=-1000, gamma=1.0, alpha =2.5)