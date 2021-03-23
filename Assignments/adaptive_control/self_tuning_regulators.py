"""
Indirect and Direct Self-Tuning Regulators

The function `simulate` considers the plant tf of 1/(M_e s^2 + Bs)
Setting `has_friction=True` simulates the continuous plant ^^ but with Coulomb friction of the type '-Asc*sat(kf*y)'

Indirect and Direct STRs are developed for either case, using
Regressors := [-y(t-1), -y(t-2), u(t-1), u(t-2)]
Parameters (ISTR) := [a1, a2, b0, b1]
Parameters (DSTR) := [-s0_tilde, -s1_tilde, r0_tilde, r1_tilde]
"""

from scipy import signal
import numpy as np
from parameter_id_classes import Estimator_RLS
from recursive_lse import measure
from matplotlib import pyplot as plt
from copy import deepcopy

EPSILON = 1e-9  # Threshold value for estimated b_0 to avoid divide-by-0


def column(vector):
    """
    Recast 1d array into into a column vector
    """
    vector = np.array(vector)
    return vector.reshape(len(vector), 1)


def controller_istr_zc(_regressors, est_params, mod_params, obs_params):
    """
    ISTR Controller when process zero is cancelled
    _uc_m1 is "u_c at time step k minus 1" and so on
    obs_params is not used
    """
    _uc, _uc_m1, _u_m1, _y, _y_m1 = _regressors
    r_1 = est_params[3]/est_params[2]
    t_0 = mod_params[2]/est_params[2]
    t_1 = mod_params[3]/est_params[2]
    s_0 = (mod_params[0]-est_params[0])/est_params[2]
    s_1 = (mod_params[1]-est_params[1])/est_params[2]

    return float(t_0*_uc + t_1*_uc_m1
                 - r_1*_u_m1 - s_0*_y
                 - s_1*_y_m1)


def controller_istr(_regressors, est_params, mod_params, obs_params):
    """
    ISTR Controller when process zero is not cancelled
    Notes:
            In this method, we don't cancel out the process zero because corresponding zero is very lightly damped
            in Z-domain. So we need to modify the reference model (we retain the poles but the observer polynomial
            (A_0) is chosen to have a zero at -100. In 'indirect-zc' the process zero is cancelled.
     """
    _uc, _uc_m1, _u_m1, _y, _y_m1 = _regressors
    a1, a2, b0, b1 = est_params.T[0]
    b0_plus_b1 = max(abs(b0 + b1), EPSILON)
    bm_pr = float((1 + mod_params[0] + mod_params[1]) / b0_plus_b1)
    mod_params[:] = np.array([mod_params[0], mod_params[1], [bm_pr*b0], [bm_pr*b1]])

    am1, am2, bm0, bm1 = mod_params.T[0]
    a01 = obs_params[0]

    t_0 = bm_pr
    t_1 = bm_pr*a01

    rs_vector = np.linalg.inv(
        np.array([[1, b0, 0], [a1, b1, b0], [a2, 0, b1]])
    ) @ np.array([[a01 + am1 - a1], [am1*a01 + am2 - a2], [am2*a01]])
    r_1, s_0, s_1 = rs_vector.T[0]

    return float(t_0*_uc + t_1*_uc_m1
                 - r_1*_u_m1 - s_0*_y
                 - s_1*_y_m1)


def controller_istr_dr(_regressors, est_params, mod_params, obs_params):
    """
    ISTR Controller when process zeros are not cancelled, and disturbance-rejection is used.
    (Disturbance is assumed to be piecewise-constant)
    """
    if len(_regressors) == 7:
        _uc, _uc_m1, _u_m1, _u_m2, _y, _y_m1, _y_m2 = _regressors
    else:
        return controller_istr(_regressors, est_params, mod_params, obs_params)

    a1, a2, b0, b1 = est_params.T[0]
    b0_plus_b1 = b0 + b1
    if abs(b0_plus_b1) < EPSILON:
        return 0.0

    bm_pr = float((1 + mod_params[0] + mod_params[1]) / b0_plus_b1)
    mod_params[:] = np.array([mod_params[0], mod_params[1], [bm_pr*b0], [bm_pr*b1]])

    am1, am2, bm0, bm1 = mod_params.T[0]
    a01 = obs_params[0]

    t_0 = bm_pr
    t_1 = bm_pr*a01

    rs0_vector = np.linalg.inv(
        np.array([[1, b0, 0], [a1, b1, b0], [a2, 0, b1]])
    ) @ np.array([[a01 + am1 - a1], [am1*a01 + am2 - a2], [am2*a01]])
    r01, s00, s01 = rs0_vector.T[0]

    y0 = -(1+r01)/b0_plus_b1
    r_1 = r01 + y0*b0
    r_2 = y0*b1
    s_0 = s00 - y0
    s_1 = s01 - y0*a1
    s_2 = -y0*a2

    return float(t_0*_uc + t_1*_uc_m1
                 - r_1*_u_m1 - r_2*_u_m2 - s_0*_y
                 - s_1*_y_m1 - s_2*_y_m2)


def controller_dstr(_regressors, est_params, mod_params, obs_params):
    """
    Direct STR Controller with the process zero cancelled
    Obs params is not used (because we cancel process zeros)
    """
    _uc, _uc_m1, _u_m1, _y, _y_m1 = _regressors
    if np.abs(est_params[2]) < EPSILON:
        return 0.0

    return float(((mod_params[2])*_uc + (mod_params[3])*_uc_m1 + (-est_params[3])*_u_m1
                 + (est_params[0])*_y + (est_params[1])*_y_m1)/est_params[2])


def square_wave(t, period=0.6, amplitude=0.2):
    if not int(t/period) % 2:
        return amplitude
    else:
        return 0.0


def plant_sim(y, y_dot, u, params, duration, dt_plant):
    M_e, B, A_sc, kf = params

    for i in range(0, int(duration/dt_plant)):
        sat_kf_y = np.clip(kf*y_dot, -1, 1)
        y_ddot = -(B/M_e)*y_dot - (A_sc/M_e)*sat_kf_y + u/M_e
        y_dot = y_dot + dt_plant*y_ddot
        y = y + dt_plant*y_dot

    return y, y_dot


def simulate(total_time=2.0, case=1, method='direct', plant_model='discrete', has_friction=False,
             DEBUGGING=False, dt=None, P_init=5000, eig_limit = 10000, f_factor=1.0):
    """
    Simulation of plant and controller in the absence and presence of Coulomb friction
    :param case: Choice of system parameters 1 or 2
    :param method: "indirect", "indirect-zc", "indirect-dr" or "direct",
                    'indirect-zc' is ISTR with all process zeros cancelled
    :param plant_model: "discrete" or "continuous"
    :return: Python dict
    """

    # Discretization time-period
    dt = 1.0/2000 if dt is None else dt
    dt_plant = 1.0/100000 if plant_model == "continuous" else dt
    assert np.isclose(dt % dt_plant, 0, atol=0.0001)

    # Choice of system parameters is broken down into cases
    if case == 1:
        M_e = 0.025
        B = 0.1
        A_sc = 0.1
    elif case == 2:
        M_e = 0.085
        B = 0.35
        A_sc = 0.15
    else:
        raise ValueError

    if not has_friction:
        A_sc = 0.0

    kf = 1000

    # Model parameters
    n_freq = 15
    damping = 1

    # Discretized Transfer Functions
    TF = signal.cont2discrete(([0, 0, 1], [M_e, B, 0]), dt=dt)
    # ^Used for controller design, and to simulate the plant if plant_model == "discrete"
    ZPK = signal.tf2zpk(TF[0], TF[1])
    TF_model = signal.cont2discrete(([0, 0, n_freq**2], [1, 2*damping*n_freq, n_freq**2]), dt=dt)
    ZPK_model = signal.tf2zpk(TF_model[0], TF_model[1])

    b0 = TF[0][0][1]
    b1 = TF[0][0][2]
    a1 = TF[1][1]
    a2 = TF[1][2]
    bm0 = TF_model[0][0][1]
    bm1 = TF_model[0][0][2]
    am1 = TF_model[1][1]
    am2 = TF_model[1][2]
    params = np.array([[a1], [a2], [b0], [b1]])
    obs_params = None
    TF_est0 = signal.cont2discrete(([0, 0, 1], [0.055, 0.225, 0]), dt=dt)

    if method == "indirect" or "indirect-dr":
        _A0 = signal.cont2discrete(([1, 100], [1, 0, 1]), dt=dt)
        a01 = -1*signal.tf2zpk(_A0[0][0], _A0[1])[0][0]
        obs_params = [a01]
        bm_pr = float((1 + am1 + am2) / (TF_est0[0][0][1] + TF_est0[0][0][2]))
        bm0 = bm_pr * TF_est0[0][0][1]
        bm1 = bm_pr * TF_est0[0][0][2]

        if method == "indirect":
            print("Using indirect STR without zero-cancellation")
            if has_friction:
                print("This controller does not have disturbance-rejection. Use 'indirect-dr' instead. "
                      "Also, you might want to increase P_init.")
            controller = controller_istr
        else:
            print("Using indirect STR without zero-cancellation, with rejection of piecewise-constant disturbances.")
            if not has_friction:
                print("Disturbance (friction) has been turned off. This controller is going to give you troubles.")
                raise ValueError
            controller = controller_istr_dr
        # Note: controller_istr and _istr_dr modify mod_params in place...

    elif method =="indirect-zc":
        print("Using indirect STR with zero-cancellation.")
        controller = controller_istr_zc
    elif method == "direct":
        print("Using direct STR with zero-cancellation.")
        controller = controller_dstr
    else:
        raise ValueError

    mod_params = np.array([[am1], [am2], [bm0], [bm1]])

    # Initialization
    if method == "direct":
        est_params = column([TF_est0[1][1] - am1, TF_est0[1][2] - am2, TF_est0[0][0][1], TF_est0[0][0][2]])
    else:
        est_params = column([TF_est0[1][1], TF_est0[1][2], TF_est0[0][0][1], TF_est0[0][0][2]])

    estimator = Estimator_RLS(P=P_init * np.identity(4), params=est_params, f_factor=f_factor, eig_limit=eig_limit)
    u_0 = controller((square_wave(0.0), 0.0, 0.0, 0.0, 0.0), est_params, mod_params, obs_params)
    regressors_1 = np.array([[-0.0], [-0.0], [u_0], [0.0]])
    y_1 = measure(regressors_1, params)
    u_1 = controller((square_wave(dt), square_wave(0.0), u_0, y_1, 0.0), est_params, mod_params, obs_params)
    regressors = column([-y_1, -0.0, u_1, u_0])
    regressors_m1 = column([-0.0, -0.0, u_0, 0.0])
    control_input = u_1

    if plant_model == "continuous":
        y, y_dot = plant_sim(0.0, 0.0, u_0, (M_e, B, A_sc, kf), duration=dt, dt_plant=dt_plant)
    elif plant_model == "discrete":
        y = y_1
        if has_friction:
            print("Coulomb friction has only been implemented for continuous plant model. Terminating...")
            raise NotImplementedError
    else:
        raise ValueError

    # For Plotting
    times = [0, dt]
    measurement_history = [0.0, y_1]
    reference_outputs = [0.0, measure(column([0.0, 0.0, square_wave(0.0), 0.0]), mod_params)]
    reference_inputs = [square_wave(dt), 0.0]
    controller_history = [u_0, u_1]
    estimate_history = [0.0, estimator.regress(regressors_1)]
    parameter_history = [estimator.params, estimator.params]

    # Iterations ----------------------------------------------------------------------------------------------------- #

    for k in range(2, int(total_time/dt)):
        times.append(k*dt)

        if plant_model == "discrete":
            y = measure(regressors, params, noise_cov=0.0)
        else:
            y, y_dot = plant_sim(y, y_dot, control_input, (M_e, B, A_sc, kf), duration=dt, dt_plant=dt_plant)

        measurement_history.append(y)

        # Estimation
        if method == "direct":
            estimator.update_P(regressors)
            y_dstr = y + am1 * measurement_history[-2] + am2 * measurement_history[-3]
            estimator.update_params(regressors, y_dstr)
            estimate_history.append(estimator.regress(regressors))  # Not exactly relevant!
            parameter_history.append([-1 * estimator.params[0], -1 * estimator.params[1],
                                      estimator.params[2], estimator.params[3]])
            controller_regressors = (square_wave(k * dt), square_wave((k - 1) * dt), controller_history[-1],
                                     measurement_history[-1], measurement_history[-2])

        elif method == "indirect-dr":
            estimator.update_P(regressors-regressors_m1)
            estimator.update_params(regressors-regressors_m1, y-measurement_history[-2])
            # estimator.update_P(regressors)
            # estimator.update_params(regressors, y)
            estimate_history.append(estimator.regress(regressors))
            parameter_history.append(estimator.params)
            controller_regressors = (square_wave(k * dt), square_wave((k - 1) * dt), controller_history[-1],
                                     controller_history[-2], measurement_history[-1], measurement_history[-2],
                                     measurement_history[-3])
        else:
            estimator.update_P(regressors)
            estimator.update_params(regressors, y)
            estimate_history.append(estimator.regress(regressors))
            parameter_history.append(estimator.params)
            controller_regressors = (square_wave(k * dt), square_wave((k - 1) * dt), controller_history[-1],
                                     measurement_history[-1], measurement_history[-2])

        # For next time-step
        control_input = controller(controller_regressors, estimator.params, mod_params, obs_params)
        regressors_m1 = deepcopy(regressors)
        regressors[1] = regressors_m1[0]
        regressors[3] = regressors_m1[2]
        regressors[0] = -1*y
        regressors[2] = control_input
        controller_history.append(control_input)

        mod_regressors = column([-1*reference_outputs[-1], -1*reference_outputs[-2],
                                 square_wave((k-1)*dt), square_wave((k-2)*dt)])
        reference_inputs.append(square_wave((k-1)*dt))
        reference_outputs.append(measure(mod_regressors, mod_params))

        if DEBUGGING:
            import pdb; pdb.set_trace()

    if method == "direct":
        true_params = column([am1-a1, am2-a2, b0, b1])
    else:
        true_params = params

    history = {"Measurements": measurement_history, "Estimates": estimate_history, "Parameters": parameter_history,
               "ControlInputs": controller_history, "TrueParameters": true_params, "Method": method,
               "ReferenceOutputs": reference_outputs, "ReferenceInputs": reference_inputs,
               "Time": times
               }

    return history


def plot_results(history, measurements=True, errors=True, controls=True, params=True,
                 xlim_params=None, ylim_params=None):
    if measurements:
        plt.plot(history["Time"], history["Measurements"], color='black', label="System State ($y$)")
        plt.plot(history["Time"], history["ReferenceOutputs"], 'r--', label="Ref. Model Output ($y_m$)")
        plt.plot(history["Time"], history["ReferenceInputs"], 'g-', label="Ref. Model Input ($u_m$)")

        plt.xlabel("Time")
        plt.legend()
        plt.show()

    if errors:
        plt.plot(history["Time"], [y - ym for y,ym in zip(history["Measurements"], history["ReferenceOutputs"])],
                 color='black', label=r"Tracking Error ($y - y_m$)")
        plt.axhline(0.0, linestyle="--", color='black')
        plt.ylabel(r"Tracking Error ($y - y_m$)")
        plt.xlabel("Time")
        plt.show()

    if controls:
        plt.plot(history["Time"], history["ControlInputs"], color="blue")
        plt.axhline(0.0, linestyle="--", color='black')
        plt.ylabel("Control Input")
        plt.xlabel("Time")
        plt.show()

    if params:
        parameters = history["Parameters"]
        true_params = history["TrueParameters"]
        colors = ["red", "orange", "blue", "green"]
        labels = []
        for i in range(np.size(history["Parameters"][0])):
            labels.append("Estimate of ")

        if history["Method"] == "indirect":
            labels[0] += r"$a_1$"
            labels[1] += r"$a_2$"
            labels[2] += r"$b_0$"
            labels[3] += r"$b_1$"

        if history["Method"] == "direct":
            labels[0] += r"$\tilde{s_0}$"
            labels[1] += r"$\tilde{s_1}$"
            labels[2] += r"$\tilde{r_0}$"
            labels[3] += r"$\tilde{r_1}$"

        for i in range(np.size(history["Parameters"][0])):
            plt.plot(history["Time"], [float(ps[i]) for ps in parameters], color=colors[i], label=labels[i])
            plt.axhline(true_params[i], linestyle="--", color=colors[i])

        if xlim_params:
            plt.xlim(0, xlim_params)
        if ylim_params:
            plt.ylim(ylim_params[0], ylim_params[1])

        plt.legend()
        plt.show()
