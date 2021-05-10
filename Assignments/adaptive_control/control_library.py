import scipy.stats as ss
from scipy import signal
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

# Values of Me, B, A_sc, Kf, Acog1, Acog3 of linear motor used for homework problems
CASES = {
    1: [0.025, 0.1, 0.1, 1000, 0.01, 0.05],
    2: [0.085, 0.35, 0.15, 1000, 0.05, 0.05],
    "Min": [0.025, 0.1, 0.1, None, 0.01, 0.01],
    "Max": [0.085, 0.35, 0.15, None, 0.05, 0.05]
         }


class StateSpacePlant:
    def __init__(self, A, B, dt, x_0 = None):
        self.state = column(np.zeros(len(A))) if not x_0 else deepcopy(x_0)
        self.derivative = column(np.zeros(len(A)))
        self.A = np.array(deepcopy(A))
        self.B = np.array(deepcopy(B))
        self.dt = dt
        self.time = 0.0
        self.outputs = []
        self.inputs = []

    def iterate(self, duration, control_input):
        self.time += duration
        for _ in range(0, int(duration / self.dt)):
            self.derivative = self.A @ self.state + self.B @ control_input
            self.state += self.derivative * self.dt

        self.outputs.append(self.state)
        self.inputs.append(control_input)


class SecondOrderPlant_Continuous:
    def __init__(self, num, den, dt, y_0=0.0, y_dot_0=0.0):
        assert len(num) == 3 and len(den) == 3
        assert num[0] == 0 and num[1] == 0

        self.num, self.den = (num, den)
        self.y = y_0
        self.y_dot = y_dot_0
        self.y_ddot = None
        self.dt = dt
        self.time = 0.0
        self.control_input = 0.0

    def iterate(self, duration, control_input):
        self.time += duration
        self.control_input = control_input
        for _ in range(0, int(duration/self.dt)):
            self.y_ddot = self.get_y_ddot(self.control_input)
            self.y_dot += self.dt*self.y_ddot
            self.y += self.dt*self.y_dot

    def get_y_ddot(self, control_input):
        return (-self.den[1]*self.y_dot - self.den[2]*self.y + self.num[2]*control_input) / self.den[0]


class Linear_Motor_Continuous(SecondOrderPlant_Continuous):
    def __init__(self, params, dt, y_0=0.0, y_dot_0=0.0, cogging=False, disturbance_type="zero"):
        self.M_e, self.B, self.A_sc, self.kf = params[0:4]
        self.Acog1, self.Acog3 = 0.0, 0.0
        self.wcog1, self.wcog3 = 0.0, 0.0
        if cogging:
            self.Acog1, self.Acog3 = params[4:6]
            self.wcog1, self.wcog3 = 2*np.pi/0.06, 6*np.pi/0.06

        if disturbance_type == "zero":
            self.disturbance = self.zero_disturbance
        elif disturbance_type == "static":
            self.disturbance = self.static_disturbance
        elif disturbance_type == "offset":
            self.disturbance = self.constant_offset_disturbance
        else:
            raise ValueError("Invalid disturbance type!")

        super().__init__(num=[0.0, 0.0, 1.0/self.M_e], den=[1.0, self.B/self.M_e, 0.0], dt=dt, y_0=y_0, y_dot_0=y_dot_0)

    def get_y_ddot(self, control_input):
        sat_kf_y = np.clip(self.kf * self.y_dot, -1, 1)
        return (super().get_y_ddot(control_input) - self.A_sc * sat_kf_y/self.M_e
                + self.Acog1*np.sin(self.wcog1*self.y)/self.M_e + self.Acog3*np.sin(self.wcog3*self.y)/self.M_e
                + self.disturbance()/self.M_e)

    def zero_disturbance(self):
        return 0.0

    def static_disturbance(self):
        return 1.0

    def constant_offset_disturbance(self):
        return 1.0 + (-1.0)**round(10*np.sin(2*self.time))


def column(vector):
    """
    Recast 1d array into into a column vector
    """
    vector = np.array(vector)
    return vector.reshape(len(vector), 1)


def measure(regressors, params, noise_cov=0.0):
    """
    :param regressors: [y(t-1), u(t-1), u(t-2)] as Numpy array
    :param params: [a, b_1, b_2]
    :param noise_cov: noise covariance
    :return: y(t)
    """
    return float(regressors.T @ params + noise(noise_cov))


def noise(noise_cov):
    return ss.multivariate_normal.rvs(cov=noise_cov)


def discretize_linear_motor(continuous_TF, dt):
    TF = signal.cont2discrete(continuous_TF, dt=dt)
    return TF[0][0][1], TF[0][0][2], TF[1][1], TF[1][2]


def square_wave_continuous(t, period=0.6, amplitude=0.2):
    if not int(t/period) % 2:
        return amplitude
    else:
        return 0.0


def square_wave_discrete(t, period, amplitude=1.0):
    if t%period<period*0.5:
        return amplitude
    else:
        return 0.0


def square_wave_discrete2sided(t, period, amplitude=1.0):
    if t % period < period * 0.5:
        return amplitude * 0.5
    else:
        return -amplitude * 0.5


def check_pe_condition_sw(order=4, time=800, sqwave_period=100, sqwave_amplitude=5.0):
    # Regressors used to check Persistent Excitation condition :-
    pe_vectors = [[[square_wave_discrete2sided(i, sqwave_period, sqwave_amplitude)] for i in range(order)][::-1]]
    pe_min = 10000.0
    pe_max = 0.0

    for t in range(order, time):
        # Make a copy of first vector
        pe_vectors.insert(0, (deepcopy(pe_vectors[0])))

        # Shift & update first vector
        pe_vectors[0].pop()
        pe_vectors[0].insert(0, [square_wave_discrete2sided(t, period=sqwave_period, amplitude=sqwave_amplitude)])

    pe_eigvals = np.linalg.eigvals(sum([np.array(r) @ np.array(r).T for r in pe_vectors]))
    pe_min = np.min([np.min(pe_eigvals), pe_min])
    pe_max = np.max([np.max(pe_eigvals), pe_max])

    print(f"Minimum eigenvalue of PE matrix was {pe_min}, max eigenvalue was {pe_max}.")


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
        colors = ["red", "orange", "blue", "green", "purple", "yellow"]
        labels = []
        for i in range(np.size(history["Parameters"][0])):
            labels.append("Estimate of ")

        if history["Method"] == "indirect":  # Indirect STR controller
            labels[0] += r"$a_1$"
            labels[1] += r"$a_2$"
            labels[2] += r"$b_0$"
            labels[3] += r"$b_1$"

        if history["Method"] == "direct":  # Direct STR controller
            labels[0] += r"$\tilde{s_0}$"
            labels[1] += r"$\tilde{s_1}$"
            labels[2] += r"$\tilde{r_0}$"
            labels[3] += r"$\tilde{r_1}$"

        if history["Method"] == "MRAC with full_state_feedback":
            labels[0] += r"$M_e$"
            labels[1] += r"$B$"
            labels[2] += r"$A_{sc}$"

        if history["Method"] == "MRAC with output_feedback":
            labels[0] += r"$r'_1$"
            labels[1] += r"$s_0$"
            labels[2] += r"$s_1$"
            labels[3] += r"$t_0$"
            labels[4] += r"$t_1$"
            labels[5] += r"$b_0$"

        if history["Method"] == "adaptive_control" or "deterministic_robust" or "deterministic_adaptive_robust":
            labels[0] += r"$M_e$"
            labels[1] += r"$B$"
            labels[2] += r"$A_{sc}$"
            labels[3] += r"$A_{cog1}$"
            labels[4] += r"$A_{cog3}$"
            labels[5] += r"$d_{0}$"

        for i in range(np.size(history["Parameters"][0])):
            plt.plot(history["Time"], [float(ps[i]) for ps in parameters], color=colors[i], label=labels[i])
            if true_params:
                plt.axhline(true_params[i], linestyle="--", color=colors[i])

        if xlim_params:
            plt.xlim(0, xlim_params)
        if ylim_params:
            plt.ylim(ylim_params[0], ylim_params[1])

        plt.legend()
        plt.show()
