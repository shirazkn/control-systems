'''
Implementation of Various Parameter ID Algorithms
Estimating parameters of system y(t) + a y(t-1) = b0 u(t-1) + b1 u(t-2) + b2 u(t-3) + c1 e(t) + c2 e(t-1)
on-line in the presence of (i) white and (ii) colored noise

NOTE:
Use `%load_ext autoreload; %autoreload 2` to turn auto-reload on in iPython!
'''

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import linear_regression_classes as classes
from control_library import noise, square_wave_discrete2sided, noise


Estimators = {
    "Equation_Error": classes.Estimator_RLS_EE,
    "ARMAX": classes.Estimator_ARMAX,
    "ARMAX_5Params": classes.Estimator_ARMAX5
}


def measure(regressors, params, noise_regressors, noise_params):
    """
    :param regressors: [y(t-1), u(t-1), u(t-2), u(t-3)] as Numpy array
    :param params: [a, b_0, b_1, b_2]
    :param noise_regressors: [e(t), e(t-1)]
    :param noise_params: [c1, c2]
    :return: y(t)
    """
    noise_regressors[1] = deepcopy(noise_regressors[0])
    noise_regressors[0] = [noise(noise_cov=1.0)]
    return float(regressors.T @ params + noise_regressors.T @ noise_params)


def simulate(estimator_type=None, simulation_case=None, total_time=200, input_period=100, input_amp=1.0):
    # params =         [ a,     b0,    b1,    b2 ]'
    params = np.array([[-0.5], [1.0], [2.0], [3.0]])

    if simulation_case == 1:
        noise_params = np.array([[0.0], [0.0]])
        estimator_type = "Equation_Error" if estimator_type is None else estimator_type
    elif simulation_case == 2:
        noise_params = np.array([[1.0], [0.0]])
        estimator_type = "Equation_Error" if estimator_type is None else estimator_type
    elif simulation_case == 3:
        noise_params = np.array([[1.0], [0.5]])
        estimator_type = "ARMAX" if estimator_type is None else estimator_type
    else:
        assert 0

    # Initialization
    regressors = np.array([[0.0], [square_wave_discrete2sided(0, input_period, input_amp)], [0.0], [0.0]])
    noise_regressors = np.array([[noise(noise_cov=1.0)], [0.0]])
    estimator = Estimators[estimator_type]()

    print(f"Using {estimator_type} model with simulation parameters c1={noise_params[0]} "
          f"and c2={noise_params[1]} (Case {simulation_case}).")

    measurement_history = [0.0]
    estimate_history = [estimator.regress()]
    parameter_history = [estimator.params]

    for t in range(1, total_time):
        y_t = measure(regressors, params, noise_regressors, noise_params)
        measurement_history.append(y_t)

        # Estimation
        estimator.update(y_t, regressors)
        estimate_history.append(estimator.regress())
        parameter_history.append(estimator.params)

        # For next time-step
        regressors[0] = -1*y_t
        regressors[3] = regressors[2]
        regressors[2] = regressors[1]
        regressors[1] = square_wave_discrete2sided(t, input_period, input_amp)

    history = {"Measurements": measurement_history, "Estimates": estimate_history, "Parameters": parameter_history,
               "True_Parameters": params, "True_Noise_Parameters": noise_params, "Estimator_Type": estimator_type}
    return history


def plot_estimates(history):
    plt.plot(history["Measurements"], 'r--', label="True Measurement ($y(t)$)")
    plt.plot(history["Estimates"], color='black', label="Estimate ($\hat y(t)$)")
    plt.title("Measurement History")
    plt.xlabel("Time-step (t)")
    plt.legend()
    plt.show()
    return


def plot_parameters(history):
    cs = ['blue', 'red', 'green', 'purple']
    param_his = history["Parameters"]
    a_vals = [params[0] for params in param_his]
    b_0_vals = [params[1] for params in param_his]
    b_1_vals = [params[2] for params in param_his]
    b_2_vals = [params[3] for params in param_his]

    plt.plot(a_vals, color=cs[0], label="Estimate of $a$")
    plt.plot(b_0_vals, color=cs[1], label="Estimate of $b_0$")
    plt.plot(b_1_vals, color=cs[2], label="Estimate of $b_1$")
    plt.plot(b_2_vals, color=cs[3], label="Estimate of $b_2$")
    plt.axhline(history["True_Parameters"][0], color=cs[0], linestyle='--', linewidth=0.5)
    plt.axhline(history["True_Parameters"][1], color=cs[1], linestyle='--', linewidth=0.5)
    plt.axhline(history["True_Parameters"][2], color=cs[2], linestyle='--', linewidth=0.5)
    plt.axhline(history["True_Parameters"][3], color=cs[3], linestyle='--', linewidth=0.5)

    if history["Estimator_Type"] == "ARMAX":
        plt.plot([params[4] for params in param_his], color='yellow', label="Estimate of $c_1$")
        plt.plot([params[5] for params in param_his], color='orange', label="Estimate of $c_2$")
        plt.axhline(history["True_Noise_Parameters"][0], color='yellow', linestyle='--', linewidth=0.5)
        plt.axhline(history["True_Noise_Parameters"][1], color='orange', linestyle='--', linewidth=0.5)

    if history["Estimator_Type"] == "ARMAX_5Params":
        plt.plot([params[4] for params in param_his], color='orange', label="Estimate of $c_2$")
        plt.axhline(history["True_Noise_Parameters"][0], color='orange', linestyle='--', linewidth=0.5)

    plt.title("Parameter History")
    plt.xlabel("Time-step (t)")
    plt.legend()
    plt.show()
    return


def plot_errors(history):
    plt.plot(np.array(history["Estimates"]) - np.array(history["Measurements"]), color='black')
    plt.title("Estimation Error History")
    plt.ylabel("Posterior Estimation Error")
    plt.xlabel("Time-step (t)")
    plt.axhline(y=0.0, linestyle='--', color='black', linewidth=0.5)
    plt.show()
