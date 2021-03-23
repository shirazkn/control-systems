import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from copy import deepcopy
import cvxpy as cvx
import dccp
from tqdm import tqdm
import matplotlib.cm as cm

n_sims = 1
cases = []
# for i in [4, 6, 8, 10, 12, 14, 16]:
#    cases.append({"N": i})

# DEBUG
cases = [{"N": 10}]

# -------------------------------------------------------------------------------------------------------------------- #
for case in cases:
    xs = []
    xhs = []
    xhts = []
    N = 10  # Time-steps between consecutive low-rate updates
    N = case["N"] if "N" in case.keys() else 10

    print(f"Simulating case N=" + str(case["N"]))
    for _ in tqdm(range(n_sims)):

        class LTV_Matrix:
            def __init__(self, matrices, modes):
                self.matrices = deepcopy(matrices)
                self.modes = deepcopy(modes)

            def get(self, t):
                return self.matrices[self.modes[t]]


        def column(vector):
            """
            Recast 1d array into into a column vector
            """
            vector = np.array(vector)
            return vector.reshape(len(vector), 1)


        def noise(cov):
            return column(multivariate_normal.rvs(cov=cov))


        total_time = 165
        total_time_simulation = total_time
        if "N" in case.keys():
            total_time_simulation -= max([c["N"] for c in cases])

        # switching_period = 25
        modes = [0 for _ in range(60)] + [1 for _ in range(25)] + [0 for _ in range(80)]
        # for i in range(total_time):
        #     if int(i / switching_period) % 2 == 0:
        #         modes.append(0)
        #     else:
        #         modes.append(1)

        DiscTimePeriod = 0.01
        RotRate = 3.5


        def if_LR_update(_k):
            return not bool(_k % N)

        # DEBUG
        if "No_LR" in case.keys():
            def if_LR_update(_k):
                return False

        # LTV Matrices
        A1 = np.array([
            [1.0, 0, DiscTimePeriod,   0],
            [0, 1.0,   0,  DiscTimePeriod],
            [0,   0, 1.0,  0],
            [0,   0,   0,   1.0],
        ])
        A2 = np.array([
            [1.0, 0, DiscTimePeriod,    0],
            [0, 1.0,   0,  DiscTimePeriod],
            [0,   0, np.cos(RotRate*DiscTimePeriod), -1*np.sin(RotRate*DiscTimePeriod)],
            [0,   0, np.sin(RotRate*DiscTimePeriod),   np.cos(RotRate*DiscTimePeriod)],
        ])

        V1 = np.diagflat([0.0001, 0.0001, 0.005, 0.005])

        # W1 = np.diagflat([0.000025, 0.000025])
        W1 = np.diagflat([0.0005, 0.0005])
        WL1 = np.diagflat([0.32, 0.32])

        # From https://docs.px4.io/master/en/advanced_config/tuning_the_ecl_ekf.html
        # and https://dewesoft.com/products/interfaces-and-sensors/gps-and-imu-devices/tech-specs

        C1 = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        CL1 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        A = LTV_Matrix([A1, A2], modes)
        V = LTV_Matrix([V1, V1], modes)
        W = LTV_Matrix([W1, W1], modes)
        WL = LTV_Matrix([WL1, WL1], modes)
        C = LTV_Matrix([C1, C1], modes)
        CL = LTV_Matrix([CL1, CL1], modes)

        # Initialization of Riccati Sequence
        P = [5.0*np.identity(4)]
        P_single = deepcopy(P)

        Beta = 1.0
        a_upper = max(np.hstack([np.linalg.eigvals(A1), np.linalg.eigvals(A2)]))
        a_lower = min(np.hstack([np.linalg.eigvals(A1), np.linalg.eigvals(A2)]))
        cL_upper = max(np.linalg.svd(CL1)[1])
        wL_lower = min(np.linalg.eigvals(WL1))

        # Generate Riccati sequence
        p_upper_uniform = 0.0
        p_lower_uniform = 20.0
        K = []
        KL = []
        K_single = []
        KL_single = []

        # ------------------------------------------------------------------------------------------------------------ #

        for k, mode in enumerate(modes):

            # -------------------------------------------------------------------------------------------------------- #

            if if_LR_update(k):
                # Estimation gains when low-rate
                Delta_inv = np.linalg.inv(np.identity(np.shape(A1)[0]) + P[k] @ (C.get(k).T @ np.linalg.inv(W.get(k))
                                                    @ C.get(k) + CL.get(k).T @ np.linalg.inv(WL.get(k)) @ CL.get(k)))
                Delta_inv_single = np.linalg.inv(np.identity(np.shape(A1)[0]) + P_single[k] @
                                                 (C.get(k).T @ np.linalg.inv(W.get(k)) @ C.get(k)
                                                  + CL.get(k).T @ np.linalg.inv(WL.get(k)) @ CL.get(k)))

                K.append(Delta_inv @ P[k] @ C.get(k).T @ np.linalg.inv(W.get(k)))
                K_single.append(Delta_inv_single @ P_single[k] @ C.get(k).T @ np.linalg.inv(W.get(k)))

                KL.append(Delta_inv @ P[k] @ CL.get(k).T @ np.linalg.inv(WL.get(k)))
                KL_single.append(Delta_inv_single @ P_single[k] @ CL.get(k).T @ np.linalg.inv(WL.get(k)))

                # P update
                F = np.identity(np.shape(A1)[0]) - K[k]@C.get(k) - KL[k]@CL.get(k)
                KWK = K[k] @ W.get(k) @ K[k].T + KL[k] @ WL.get(k) @ KL[k].T
                P.append(A.get(k) @ ((F @ P[k] @ F.T) + KWK) @ A.get(k).T + V.get(k))

                F_single = np.identity(np.shape(A1)[0]) - K_single[k]@C.get(k) - KL_single[k]@CL.get(k)
                KWK_single = K_single[k] @ W.get(k) @ K_single[k].T + KL_single[k] @ WL.get(k) @ KL_single[k].T
                P_single.append(A.get(k) @ ((F_single @ P[k] @ F_single.T) + KWK_single) @ A.get(k).T + V.get(k))

            # -------------------------------------------------------------------------------------------------------- #

            else:
                # Estimation gains when no low-rate
                Delta_inv = np.linalg.inv(np.identity(np.shape(A1)[0]) + P[k] @ C.get(k).T
                                          @ np.linalg.inv(W.get(k)) @ C.get(k))
                K.append(Delta_inv @ P[k] @ C.get(k).T @ np.linalg.inv(W.get(k)))
                KL.append(None)
                K_single.append(None)
                KL_single.append(None)

                # P update
                F = np.identity(np.shape(A1)[0]) - K[k] @ C.get(k)
                KWK = K[k] @ W.get(k) @ K[k].T
                P.append(A.get(k) @ ((F @ P[k] @ F.T) + KWK) @ A.get(k).T + V.get(k))

                P_single.append(A.get(k) @ P[k] @ A.get(k).T + V.get(k))

            # -------------------------------------------------------------------------------------------------------- #

            if if_LR_update(k) and k > 40:
                p_lower_uniform = min(min(np.linalg.eigvals(P[k])), p_lower_uniform)
                p_upper_uniform = max(max(np.linalg.eigvals(P[k])), p_upper_uniform)

        # ------------------------------------------------------------------------------------------------------------ #

        P.pop();  # Remove extra entry at the end
        P_single.pop();  # Remove extra entry at the end

        #  Initialization for simulation
        x_hat = [np.array([[0.5], [1.0], [10.0], [10.0]])]
        x_hat_single = deepcopy(x_hat)
        x_hat_tilde = deepcopy(x_hat)
        x_hat_tilde_single = deepcopy(x_hat)
        x = [np.array([[0.0], [0.0], [15.0], [15.0]])]
        y = []
        yL = []
        yL_tilde = []
        Da = []


        def state_transition(x, k):
            return A.get(k) @ x + noise(V.get(k))


        def est_transition(xh, k, y, yL=None):
            if yL is None:
                return A.get(k) @ xh + A.get(k) @ K[k]  @ (y - C.get(k) @ xh)
            else:
                return (A.get(k) @ xh + A.get(k) @ K[k] @ (y - C.get(k) @ xh)
                        + A.get(k) @ KL[k] @ (yL - CL.get(k) @ xh))


        def measurement(x, k):
            return C.get(k) @ x + noise(W.get(k))


        def measurementL(x, k, attack: np.array = None):
            if if_LR_update(k):
                if attack is None:
                    return CL.get(k) @ x + noise(WL.get(k))
                else:
                    return CL.get(k) @ x + noise(WL.get(k)) + attack
            else:
                return None


        def make_attack(k):
            if if_LR_update(k):
                evals, evecs = np.linalg.eig(KL[k].T @ A.get(k).T @ np.linalg.inv(P[k]) @ A.get(k) @ KL[k])
                return column(Beta*evecs[:, np.argmax(evals)])
            else:
                return None


        def make_attack_optimal(k):
            if if_LR_update(k):
                if k - N >= 0:
                    Phi_CL = np.identity(np.shape(A.get(k))[0])
                    for i in range(k+1, k+N-1):
                        Phi_CL = A.get(i) @ (np.identity(np.shape(A.get(i))[0]) - K[i] @ C.get(i)) @ Phi_CL
                    A_CL_k = A.get(k) @ (np.identity(np.shape(A.get(k))[0]) - K[k] @ C.get(k) - KL[k] @ CL.get(k))
                    AKD = A.get(k) @ KL[k]

                    a = cvx.Variable(2)
                    _dele = x_hat_tilde[k] - x_hat[k]
                    obj = cvx.norm(Phi_CL @ (AKD @ a + (A_CL_k @ _dele)[:, 0]))
                    opt_prob = cvx.Problem(cvx.Maximize(obj), [cvx.norm(a) <= Beta])
                    opt_prob.solve(method='dccp')
                    return column(a.value)
                else:
                    make_attack(k)
            else:
                return None


        # Simulation
        for k, mode in enumerate(modes[:total_time_simulation]):
            y.append(measurement(x[k], k))
            yL.append(measurementL(x[k], k, attack=None))
            Da.append(make_attack_optimal(k))
            yL_tilde.append(measurementL(x[k], k, attack=Da[k]))

            # Set <k+1>th entry
            x_hat.append(est_transition(x_hat[k], k, y[k], yL[k]))
            x_hat_tilde.append(est_transition(x_hat_tilde[k], k, y[k], yL_tilde[k]))
            x.append(state_transition(x[k], k))


        alpha = 0.1
        dele = x_hat[0] - x_hat_tilde[0]
        V = dele.T @ np.linalg.inv(P[0]) @ dele

        p_lower = p_lower_uniform
        p_upper = p_upper_uniform


        def get_e_star():
            gamma = alpha * p_lower ** 2 / (p_upper ** 2 * (1 - alpha))
            return a_upper * Beta * cL_upper * p_upper ** 1.5 * (1 + np.sqrt(1 + gamma)) / (
                        wL_lower * np.sqrt(p_lower * alpha * gamma))


        # Get Lyapunov sequence and obtain bound on alpha
        e_star = [get_e_star()]
        P_eigvals = np.array([])

        for k, mode in enumerate(modes[:total_time_simulation]):
            if if_LR_update(k) and k > 0:
                dV = dele.T @ np.linalg.inv(P[k]) @ dele - V

                if np.linalg.norm(dele) > 0:
                    assert dV < 0
                    alpha = -dV/V

                e_star.append(get_e_star())

                # For next time-step
                p_lower = np.min(np.linalg.eigvals(P[k]))
                p_upper = np.max(np.linalg.eigvals(P[k]))
                dele = x_hat[k] - x_hat_tilde[k]
                V = dele.T @ np.linalg.inv(P[k]) @ dele
                dele = A.get(k) @ (np.identity(np.shape(A.get(k))[0]) - K[k] @ C.get(k) - KL[k] @ CL.get(k)) @ dele

            else:
                dele = A.get(k) @ (np.identity(np.shape(A.get(k))[0]) - K[k] @ C.get(k)) @ dele
                e_star.append(max(np.linalg.eigvals(A.get(k) @ (np.identity(np.shape(A.get(k))[0]) - K[k] @ C.get(k))))
                              * e_star[-1])

        e_star[0] = e_star[1]
        xs.append(x)
        xhs.append(x_hat)
        xhts.append(x_hat_tilde)

    errors = []
    errors_af = []
    for sim in range(n_sims):
        errors.append([np.linalg.norm(xht - x) for xht, x in zip(xhts[sim], xs[sim])])
        errors_af.append([np.linalg.norm(xh - x) for xh, x in zip(xhs[sim], xs[sim])])

    errors = np.array(errors)
    errors_af = np.array(errors_af)
    errors_avg = []
    errors_af_avg = []
    for i in range(len(errors[0])):
        errors_avg.append(np.average(errors[:, i]))
        errors_af_avg.append(np.average(errors_af[:, i]))

    biases = []
    for sim in range(n_sims):
        biases.append([np.linalg.norm(xht - xh) for xht, xh in zip(xhts[sim], xhs[sim])])

    biases = np.array(biases)
    biases_avg = []
    for i in range(len(biases[0])):
        biases_avg.append(np.average(biases[:, i]))

    case["History"] = {"Errors": errors_avg, "Errors AF": errors_af_avg, "E_star_final": e_star[-1]}
# -------------------------------------------------------------------------------------------------------------------- #

PLOT1 = True
PLOT2 = False
PLOT3 = False
PLOT4 = False
PLOT5 = True

if PLOT1:
    plt.plot([x[0] for x in x], [x[1] for x in x], linestyle="-", color="black", label="System State")
    plt.plot([x[0] for x in x_hat], [x[1] for x in x_hat], linestyle="dashed", color="green",
             label="DRKF Estimate (without Attack)")
    plt.plot([x[0] for x in x_hat_tilde], [x[1] for x in x_hat_tilde], linestyle="dotted", color="red",
             label="DRKF Estimate (with Attack)")
    plt.xlabel(r'$x_1(k)$')
    plt.ylabel(r'$x_2(k)$')
    plt.legend()
    plt.show()

if PLOT2:
    # plt.plot([np.linalg.norm(xh-x) for xh, x in zip(x_hat, x)]); plt.show()
    plt.plot([np.linalg.norm(xh-xht) for xh, xht in zip(x_hat, x_hat_tilde)], color='blue',
             label=r"Estimation Error Bias" + r" ($\|\Delta e_k\|$) ")
    plt.plot(e_star, color='red', linestyle='--', label="Theoretical bound")
    # plt.legend(prop={'family': 'Times New Roman'})
    plt.ylabel("Bias in Estimation Error")
    plt.xlabel("Time-step")
    plt.legend()
    plt.ylim(0.0, 60)
    # plt.axhline(e_star, color='red', linestyle='--', linewidth=0.5)
    plt.show()

if PLOT3:
    # plt.plot([np.linalg.norm(xh-x) for xh, x in zip(x_hat, x)]); plt.show()
    plt.plot(errors_avg, color='black', label=r"Estimation Error ($\|\tilde e_k\|$)")
    plt.plot(biases_avg, linestyle='--', color='blue', label=r"Bias in Estimation Error ($\|\Delta e_k\|$)")
    plt.legend(loc="upper right")
    # plt.legend(prop={'family': 'Times New Roman'})
    plt.ylabel("Mean Squared Estimation Error")
    plt.xlabel("Time-step")
    # plt.axhline(e_star, color='red', linestyle='--', linewidth=0.5)
    plt.ylim(0.0, 5)
    plt.show()

if PLOT4:
    e_star = []
    err = []
    err_af = []
    x_vals = []
    for case in cases:
        x_vals.append(case["N"])
        e_star.append(float(case["History"]["E_star_final"][0][0]))
        err.append(float(case["History"]["Errors"][-1]))
        err_af.append(float(case["History"]["Errors AF"][-1]))

    plt.plot(x_vals, e_star)
    plt.title("E_star_final vs N")
    plt.show()
    plt.plot(x_vals, err, label="with attack")
    # plt.plot(x_vals, err_af, label="without attack")
    plt.ylabel("Estimation Error at final time-step")
    plt.legend()
    plt.show()

if PLOT5:
    colors = cm.get_cmap("Set1").colors + cm.get_cmap("Set2").colors

    for i, case in enumerate(cases):
        plt.plot(case["History"]["Errors"], color=colors[i],
                 label="N = " + str(case["N"]))
        plt.plot(case["History"]["Errors AF"], color=colors[i], linestyle="dashed",
                 label="N = " + str(case["N"]))

    ylabel = "Estimation Error " + r"($\|e_k\|^2$)"
    plt.ylabel(ylabel)
    plt.legend()
    plt.ylim(0.0, 1.5)
    plt.show()


plt.plot(case["History"]["Errors"], color=colors[0],
         label="with attack")
plt.plot(case["History"]["Errors AF"], color=colors[1], linestyle="dashed",
         label="without attack")

ylabel = "Estimation Error " + r"($\|e_k\|^2$)"
plt.ylabel(ylabel)
plt.xlabel("Time-step")
plt.legend()
plt.ylim(0.0, 1.5)
plt.show()




