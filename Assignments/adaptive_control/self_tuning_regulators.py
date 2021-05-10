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
from copy import deepcopy

from linear_regression_classes import Estimator_RLS
from control_library import measure, column, square_wave_continuous, Linear_Motor_Continuous, discretize_linear_motor, \
    CASES

EPSILON = 1e-9  # Threshold value for estimated b_0 to avoid divide-by-0

# ------------------------------------------------------------------------------------------- #
# --------------------------------------- CONTROLLERS --------------------------------------- #
# ------------------------------------------------------------------------------------------- #


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

# -------------------------------------------------------------------------------------------- #
# ---------------------------------------- SIMULATION ---------------------------------------- #
# -------------------------------------------------------------------------------------------- #


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

    # Choice of system parameters as tuple (Me, B, A_sc, kf)
    params_cont = CASES[case]
    if not has_friction:
        params_cont[2] = 0.0

    # Model parameters
    n_freq = 15
    damping = 1

    # Discretized Transfer Functions
    
    b0, b1, a1, a2 = discretize_linear_motor(continuous_TF=([0, 0, 1],
                                                            [params_cont[0], params_cont[1], 0]), dt=dt)
    bm0, bm1, am1, am2 = discretize_linear_motor(continuous_TF=([0, 0, n_freq**2],
                                                                [1, 2*damping*n_freq, n_freq**2]), dt=dt)
    b0est, b1est, a1est, a2est = discretize_linear_motor(continuous_TF=([0, 0, 1], [0.055, 0.225, 0]), dt=dt)

    params_disc = np.array([[a1], [a2], [b0], [b1]])
    obs_params = None

    if method == "indirect" or "indirect-dr":
        _A0 = signal.cont2discrete(([1, 100], [1, 0, 1]), dt=dt)
        a01 = -1*signal.tf2zpk(_A0[0][0], _A0[1])[0][0]
        obs_params = [a01]
        bm_pr = float((1 + am1 + am2) / (b0est + b1est))
        bm0 = bm_pr * b0est
        bm1 = bm_pr * b1est

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
        est_params = column([a1est - am1, a2est - am2, b0est, b1est])
    else:
        est_params = column([a1est, a2est, b0est, b1est])

    # First two time-steps computed manually
    estimator = Estimator_RLS(P=P_init * np.identity(4), params=est_params, f_factor=f_factor, eig_limit=eig_limit)
    u_0 = controller((square_wave_continuous(0.0), 0.0, 0.0, 0.0, 0.0), est_params, mod_params, obs_params)
    regressors_1 = np.array([[-0.0], [-0.0], [u_0], [0.0]])
    y_1 = measure(regressors_1, params_disc)
    u_1 = controller((square_wave_continuous(dt), square_wave_continuous(0.0), u_0, y_1, 0.0), est_params, mod_params,
                     obs_params)
    regressors = column([-y_1, -0.0, u_1, u_0])
    regressors_m1 = column([-0.0, -0.0, u_0, 0.0])
    control_input = u_1

    if plant_model == "continuous":
        plant = Linear_Motor_Continuous(params=params_cont, dt=dt_plant)
        plant.iterate(duration=dt, control_input=u_0)
    elif plant_model == "discrete":
        if has_friction:
            print("Coulomb friction has only been implemented for continuous plant model. Terminating...")
            raise NotImplementedError
    else:
        raise ValueError

    # For Plotting
    times = [0, dt]
    measurement_history = [0.0, y_1]
    reference_outputs = [0.0, measure(column([0.0, 0.0, square_wave_continuous(0.0), 0.0]), mod_params)]
    reference_inputs = [square_wave_continuous(dt), 0.0]
    controller_history = [u_0, u_1]
    estimate_history = [0.0, estimator.regress(regressors_1)]
    parameter_history = [estimator.params, estimator.params]

    # Iterations ----------------------------------------------------------------------------------------------------- #

    for k in range(2, int(total_time/dt)):
        times.append(k*dt)

        # Simulate plant
        if plant_model == "discrete":
            y = measure(regressors, params_disc, noise_cov=0.0)
        else:
            plant.iterate(duration=dt, control_input=control_input)
            y = plant.y

        measurement_history.append(y)

        # Estimation
        if method == "direct":
            estimator.update_P(regressors)
            y_dstr = y + am1 * measurement_history[-2] + am2 * measurement_history[-3]
            estimator.update_params(regressors, y_dstr)
            estimate_history.append(estimator.regress(regressors))  # Not exactly relevant!
            parameter_history.append([-1 * estimator.params[0], -1 * estimator.params[1],
                                      estimator.params[2], estimator.params[3]])
            controller_regressors = (square_wave_continuous(k * dt), square_wave_continuous((k - 1) * dt), controller_history[-1],
                                     measurement_history[-1], measurement_history[-2])

        elif method == "indirect-dr":
            estimator.update_P(regressors-regressors_m1)
            estimator.update_params(regressors-regressors_m1, y-measurement_history[-2])
            # estimator.update_P(regressors)
            # estimator.update_params(regressors, y)
            estimate_history.append(estimator.regress(regressors))
            parameter_history.append(estimator.params)
            controller_regressors = (square_wave_continuous(k * dt), square_wave_continuous((k - 1) * dt), controller_history[-1],
                                     controller_history[-2], measurement_history[-1], measurement_history[-2],
                                     measurement_history[-3])
        else:
            estimator.update_P(regressors)
            estimator.update_params(regressors, y)
            estimate_history.append(estimator.regress(regressors))
            parameter_history.append(estimator.params)
            controller_regressors = (square_wave_continuous(k * dt), square_wave_continuous((k - 1) * dt), controller_history[-1],
                                     measurement_history[-1], measurement_history[-2])

        # For next time-step
        control_input = controller(controller_regressors, estimator.params, mod_params, obs_params)
        regressors_m1 = deepcopy(regressors)
        regressors[1] = regressors_m1[0]
        regressors[3] = regressors_m1[2]
        regressors[0] = -1*y
        regressors[2] = control_input
        controller_history.append(control_input)

        mod_regressors = column([-1 * reference_outputs[-1], -1 * reference_outputs[-2],
                                 square_wave_continuous((k - 1) * dt), square_wave_continuous((k - 2) * dt)])
        reference_inputs.append(square_wave_continuous((k - 1) * dt))
        reference_outputs.append(measure(mod_regressors, mod_params))

        if DEBUGGING:
            import pdb; pdb.set_trace()

    if method == "direct":
        params_disc = column([am1-a1, am2-a2, b0, b1])

    history = {"Measurements": measurement_history, "Estimates": estimate_history, "Parameters": parameter_history,
               "ControlInputs": controller_history, "TrueParameters": params_disc, "Method": method,
               "ReferenceOutputs": reference_outputs, "ReferenceInputs": reference_inputs,
               "Time": times
               }

    return history
