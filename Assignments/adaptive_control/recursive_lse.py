'''
Recursive Least Squares Algorithm
Estimating parameters of system y(t) + a y(t-1) = b1 u(t-1) + b2 u(t-2) + e(t)
on-line in the presence of Gaussian noise e(t), u(t) is a square wave
'''
import numpy as np
import scipy.stats as s
import matplotlib.pyplot as plt
from copy import deepcopy
from labellines import labelLine, labelLines
from parameter_id_classes import Estimator_RLS


def measure(regressors, params, noise_cov=0.0):
    """
    :param regressors: [y(t-1), u(t-1), u(t-2)] as Numpy array
    :param params: [a, b_1, b_2]
    :param noise_cov: noise covariance
    :return: y(t)
    """
    return float(regressors.T @ params + noise(noise_cov))


def noise(noise_cov):
    return s.multivariate_normal.rvs(cov=noise_cov)


def square_wave_discrete(t, period, amplitude=1.0):
    if t%period<period*0.5:
        return amplitude
    else:
        return 0.0


def simulate(estimator_init, total_time=200, noise_cov=0.0, input_period=100, input_amp=1.0):
    a = -0.5
    b_1 = 1.0
    b_2 = -1.0
    params = np.array([[a], [b_1], [b_2]])

    # Initialization
    regressors = np.array([[0.0], [square_wave_discrete(0, input_period, input_amp)], [0.0]])
    estimator = Estimator_RLS(estimator_init["P"], estimator_init["params"], estimator_init["f_factor"])
    measurement_history = [0.0]
    estimate_history = [estimator.regress(regressors)]
    parameter_history = [estimator.params]

    for t in range(1, total_time):
        y_t = measure(regressors, params, noise_cov)
        measurement_history.append(y_t)

        # Estimation
        estimator.update_P(regressors)
        estimator.update_params(regressors, y_t)
        estimate_history.append(estimator.regress(regressors))
        parameter_history.append(estimator.params)

        # For next time-step
        regressors[0] = -1*y_t
        regressors[2] = regressors[1]
        regressors[1] = square_wave_discrete(t, input_period, input_amp)

    history = {"Measurements": measurement_history, "Estimates": estimate_history, "Parameters": parameter_history}
    return history


def plot_estimates(history, axis =None):
   if not axis:
        plt.plot(history["Measurements"], 'r--', label="True Measurement ($y(t)$)")
        plt.plot(history["Estimates"], color='black', label="Estimate ($\hat y(t)$)")
        plt.title("Measurement History")
        plt.xlabel("Time-step (t)")
        plt.legend()
        plt.show()
        return
   axis.plot(history["Measurements"], 'r--', label="$y(t)$")
   axis.plot(history["Estimates"], color='black', label="$\hat y(t)$")
   return axis

def plot_parameters(history, axis = None):
    param_his = history["Parameters"]
    a_vals = [params[0] for params in param_his]
    b_1_vals = [params[1] for params in param_his]
    b_2_vals = [params[2] for params in param_his]
    if not axis:
        plt.plot(a_vals, 'b', label="Estimate of $a$")
        plt.plot(b_1_vals, 'r', label="Estimate of $b_1$")
        plt.plot(b_2_vals, 'g', label="Estimate of $b_2$")
        plt.title("Parameter History")
        plt.xlabel("Time-step (t)")
        plt.legend()
        plt.show()
        return
    axis.plot(a_vals, 'b', label=r"$\hat a$")
    axis.plot(b_1_vals, 'r', label=r"$\hat b_1$")
    axis.plot(b_2_vals, 'g', label=r"$\hat b_2$")


def plot8_wrt_initialization_1(noise=False):

    param_range = [[[-10], [0], [0]],
                   [[10], [0], [0]],
                   [[0], [-5], [-6]],
                   [[0], [6], [-5]]]
    history = []

    fig, axes = plt.subplots(4, 1, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.suptitle(r'Parameter history as a function of $\hat{\theta}_0$', y=0.92)
    for i in range(4):
        estimator_init = get_empty_dict()
        estimator_init["params"] = param_range[i]
        noise_cov = 0.2 if noise else 0.0
        history.append(simulate(estimator_init, total_time=400, noise_cov=noise_cov))

        plot_parameters(history[-1], axis=axes.flat[i])
        axes.flat[i].set_title(r'$\hat{\theta}' + f' (0)_{i}' + '=$' + f'$[{param_range[i][0]}, {param_range[i][1]}, {param_range[i][2]}]^T$', y=0.80)

    for axis in axes.flat:
        axis.label_outer()

    offset_legend(fig, [-0.15, -0.3])
    plt.show()
    plot_errors_combined([np.array(h["Measurements"][0:99]) - np.array(h["Estimates"][0:99]) for h in history],
                         [f'{i}' for i in [1, 2, 3, 4]])


def plot8_wrt_initialization_2_cov(noise=False):

    param_range = [0.1, 0.5, 4.0, 20.0]
    history = []

    fig, axes = plt.subplots(4, 1, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.suptitle(r'Parameter history as a function of P(0)$', y=0.92)
    for i in range(4):
        estimator_init = get_empty_dict()
        estimator_init["P"] = param_range[i]*np.identity(3)
        noise_cov = 0.2 if noise else 0.0
        history.append(simulate(estimator_init, total_time=400, noise_cov=noise_cov))

        plot_parameters(history[-1], axis=axes.flat[i])

    axes.flat[0].set_title(r'$P(0)_1$' + '$=$' + f'${param_range[0]}I_3$', y=0.80)
    axes.flat[1].set_title(r'$P(0)_2$' + '$=$' + f'${param_range[1]}I_3$', y=0.80)
    axes.flat[2].set_title(r'$P(0)_3$' + '$=$' + f'${param_range[2]}I_3$', y=0.80)
    axes.flat[3].set_title(r'$P(0)_4$' + '$=$' + f'${param_range[3]}I_3$', y=0.80)

    for axis in axes.flat:
        axis.label_outer()

    offset_legend(fig, [-0.15, -0.3])
    plt.show()
    plot_errors_combined([np.array(h["Measurements"][1:150]) - np.array(h["Estimates"][1:150]) for h in history],
                         [f'{i}' for i in [1, 2, 3, 4]])

def plot8_wrt_initialization_3_ffac(noise=False):
        param_range = [0.05, 0.3, 0.7, 0.9]
        history = []

        fig, axes = plt.subplots(4, 1, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
        fig.suptitle(r'Parameter history as a function of $\lambda$', y=0.92)
        for i in range(4):
            estimator_init = get_empty_dict()
            estimator_init["f_factor"] = float(param_range[i])
            estimator_init["P"] = 0.1 * np.identity(3)
            noise_cov = 0.2 if noise else 0.0
            history.append(simulate(estimator_init, total_time=400, noise_cov=noise_cov))
            axes.flat[i].set_title(r'$\lambda$' + '$=$' + f'${param_range[i]}$', y=0.80)
            plot_parameters(history[-1], axis=axes.flat[i])

        for axis in axes.flat:
            axis.label_outer()

        offset_legend(fig, [-0.15, -0.3])
        plt.show()
        history = history[2:]
        plot_errors_combined([np.array(h["Measurements"][1:150]) - np.array(h["Estimates"][1:150]) for h in history],
                             [f'{i}' for i in [3, 4]])


def plot8_wrt_initialization_4_period(noise=False):
    param_range = [10, 50, 200, 800]
    history = []

    fig, axes = plt.subplots(4, 1, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.suptitle(r'Parameter history as a function of input period', y=0.92)
    for i in range(4):
        noise_cov = 0.2 if noise else 0.0
        history.append(simulate(get_empty_dict(), total_time=400, noise_cov=noise_cov, input_period=param_range[i]))
        axes.flat[i].set_title(r'Period: ' + f'${param_range[i]}$', y=0.80)
        plot_parameters(history[-1], axis=axes.flat[i])

    for axis in axes.flat:
        axis.label_outer()

    offset_legend(fig, [-0.15, -0.3])
    plt.show()
    plot_errors_combined([np.array(h["Measurements"][1:150]) - np.array(h["Estimates"][1:150]) for h in history],
                         [f'{i}' for i in [1, 2, 3, 4]])


def offset_legend(fig, offset):
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    leg = fig.legend(lines, labels, loc='upper right')
    bb = leg.get_bbox_to_anchor().inverse_transformed(fig.axes[-1].transAxes)
    # ^ Deprecated, use transformed(transform.inverted())
    bb.x0 += offset[0]; bb.x1 += offset[0]; bb.y0 += offset[1]; bb.y1 += offset[1]
    leg.set_bbox_to_anchor(bb, transform=fig.axes[-1].transAxes)


def plot_errors_combined(errors, labels):
    for error_vals, label in zip(errors, labels):
        plt.plot(error_vals, label=label)

    plt.xlabel("Time-step")
    plt.ylabel("Posterior Estimation Error")
    labelLines(plt.gca().get_lines(), zorder=2.5)
    plt.show()

def get_empty_dict():
    return {
        "P": None,
        "params": None,
        "f_factor": None
    }