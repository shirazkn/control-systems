"""
Implementation of Adaptive (non-robust) Controller contrasted with a Deterministic Robust Controller (DRC)
for a linear motor with cogging forces, Coulomb friction and lumped external disturbances.
"""
import numpy as np
from control_library import CASES, Linear_Motor_Continuous, SecondOrderPlant_Continuous, square_wave_continuous, column
from copy import deepcopy

INIT_ESTIMATE = [0.225, 0.125, 0.03, 0.03, 0, 0.055]
TRUE_ESTIMATE = [[0.1, 0.1, 0.01, 0.05, 1.0, 0.025], [0.35, 0.15, 0.05, 0.05, 1.0, 0.085]]
REF_MODEL = (15, 1)  # Nat_freq. and damping ratio


class Controller_AC:
    """
    ADAPTIVE CONTROL
    """
    def __init__(self, init_estimate, ref_model, controller_params):
        self.ref_model = ref_model

        # [ B     Asc     Acog1      Acog3    d_0     M ]
        self.params = column(init_estimate)
        self.k1, self.k2 = controller_params[0:2]
        self.gamma = np.diag(controller_params[2:8])
        self.kf = 1000.0
        self.regressors = column(np.zeros(6))

        self.wcog1, self.wcog3 = 2*np.pi/0.06, 6*np.pi/0.06

        self.parameter_history = [
            # [ M     B     Asc     Acog1      Acog3    d_0  ] (This is ordered differently from self.params)
            deepcopy([self.params[5][0], self.params[0][0], self.params[1][0], self.params[2][0], self.params[3][0],
                      self.params[4][0]])]
        self.control_input = self.gamma[5][0] * (self.ref_model.get_y_ddot(square_wave_continuous(0.0)))
        self.time = 0.0

    def iterate(self, duration, inputs):
        self.time += duration
        y, y_dot = inputs
        ym, ym_dot = self.ref_model.y, self.ref_model.y_dot
        ym_ddot = self.ref_model.get_y_ddot(self.ref_model.control_input)
        sat_kf_y = np.clip(self.kf * y_dot, -1, 1)

        p = y_dot - ym_dot + self.k1 * (y - ym)
        self.regressors = column([-y_dot,
                                  -sat_kf_y,
                                  np.sin(self.wcog1*y),
                                  np.sin(self.wcog3*y),
                                  1.0,
                                  -(ym_ddot - self.k1*(y_dot - ym_dot)) ])

        # Update parameters
        self.params = self.params + ((self.gamma @ self.regressors)*p*duration)
        self.control_input = -float(self.regressors.T@self.params) - self.k2*p

        self.parameter_history.append(deepcopy([self.params[5][0], self.params[0][0], self.params[1][0],
                                                self.params[2][0], self.params[3][0], self.params[4][0]]))

    def get_control_input(self):
        return self.control_input


class Controller_DRC(Controller_AC):
    """
    DETERMINISTIC ROBUST CONTROL
    """
    def __init__(self, *args, **kwargs):
        self.sgn_p = 1.0
        self.scale_factor = 5.0  # Tuning parameter corresponding to approximation of signum(.) as tanh(.)
        self.params_max = [CASES["Max"][1], CASES["Max"][2], CASES["Max"][4], CASES["Max"][5], +2, CASES["Max"][0]]
        self.params_min = [CASES["Min"][1], CASES["Min"][2], CASES["Min"][4], CASES["Min"][5], -2, CASES["Min"][0]]
        self.params_range = column(self.params_max) - column(self.params_min)

        super().__init__(*args, **kwargs)

    def iterate(self, duration, inputs):
        self.time += duration
        y, y_dot = inputs
        ym, ym_dot = self.ref_model.y, self.ref_model.y_dot
        ym_ddot = self.ref_model.get_y_ddot(self.ref_model.control_input)
        sat_kf_y = np.clip(self.kf * y_dot, -1, 1)

        p = y_dot - ym_dot + self.k1 * (y - ym)
        self.regressors = column([-y_dot,
                                  -sat_kf_y,
                                  np.sin(self.wcog1 * y),
                                  np.sin(self.wcog3 * y),
                                  1.0,
                                  -(ym_ddot - self.k1 * (y_dot - ym_dot))])

        # Update parameters (Disabled for DRC)
        # self.params = self.params + ((self.gamma @ self.regressors) * p * duration)
        self.control_input = -float(self.regressors.T @ self.params) - self.k2 * p
        self.sgn_p = np.tanh(self.scale_factor * (y_dot - ym_dot + self.k1 * (y - ym)))
        self.parameter_history.append(deepcopy([self.params[5][0], self.params[0][0], self.params[1][0],
                                                self.params[2][0], self.params[3][0], self.params[4][0]]))

    def get_control_input(self):
        regressors_abs = []
        for i in range(6):
            regressors_abs.append(np.abs(self.regressors[i][0]))
        return self.control_input - float(column(regressors_abs).T @ self.params_range * self.sgn_p)


class Controller_DARC(Controller_AC):
    """
    DETERMINISTIC ADAPTIVE ROBUST CONTROL
    """
    def __init__(self, init_estimate, ref_model, controller_params, auto_tuning=False):
        self.sgn_p = 0.0
        self.h = None
        k_max = 90.0
        h_max = 7.0
        self.kappa = 0.2785
        self.eps = (self.kappa*h_max**2)/(k_max - controller_params[1])

        self.params_max = [CASES["Max"][1], CASES["Max"][2], CASES["Max"][4], CASES["Max"][5], +2, CASES["Max"][0]]
        self.params_min = [CASES["Min"][1], CASES["Min"][2], CASES["Min"][4], CASES["Min"][5], -2, CASES["Min"][0]]
        self.params_range = column(self.params_max) - column(self.params_min)
        super().__init__(init_estimate, ref_model, controller_params)
        Wd = np.diag([max(self.params_max[i] + self.params_min[i], self.params_max[i] - self.params_min[i])
                     for i in range(len(self.params))])
        Wd_sq = Wd @ Wd
        damping_ratio = 0.5
        gamma_scale = k_max**2 / (4*(damping_ratio**2)*40.0)
        self.gamma = gamma_scale*Wd_sq
        # import pdb; pdb.set_trace()

    def iterate(self, duration, inputs):
        self.time += duration
        y, y_dot = inputs
        ym, ym_dot = self.ref_model.y, self.ref_model.y_dot
        ym_ddot = self.ref_model.get_y_ddot(self.ref_model.control_input)

        p = y_dot - ym_dot + self.k1 * (y - ym)
        sat_kf_y = np.clip(self.kf * y_dot, -1, 1)

        self.regressors = column([-y_dot,
                                  -sat_kf_y,
                                  np.sin(self.wcog1 * y),
                                  np.sin(self.wcog3 * y),
                                  1.0,
                                  -(ym_ddot - self.k1 * (y_dot - ym_dot))])
        regressors_abs = []
        for i in range(6):
            regressors_abs.append(np.abs(self.regressors[i][0]))
        self.h = float(column(regressors_abs).T @ self.params_range)
        self.sgn_p = np.tanh((self.kappa*self.h/self.eps) * (y_dot - ym_dot + self.k1 * (y - ym)))

        # Update parameters
        self.params = self.params + ((self.gamma @ self.regressors) * p * duration)
        for i in range(6):
            self.params[i][0] = max(self.params[i][0], self.params_min[i])
            self.params[i][0] = min(self.params[i][0], self.params_max[i])

        self.control_input = -float(self.regressors.T @ self.params) - self.k2 * p - self.h * self.sgn_p
        self.parameter_history.append(deepcopy([self.params[5][0], self.params[0][0], self.params[1][0],
                                                self.params[2][0], self.params[3][0], self.params[4][0]]))

        # print(self.regressors.T @ self.Wd_sq @ self.regressors)

    def get_control_input(self):
        return self.control_input


def simulate(total_time=4.0, case=1, controller_type=None, friction=True, cogging=True, disturbance_type="zero",
             dt_plant=None, dt_controller=None, controller_params: tuple = None, init_estimate: list = None,
             ref_input_type="square"):
    """
    Simulates linear motor with friction and MRAC controller
    :param total_time: Total time of simulation
    :param case: Choice of system parameters (see control_library.py
    :param controller_type: ac for adaptive control,
                            drc for det. robust control
    :param friction: Simulate friction (True or False)
    :param cogging: Simulate cogging forces (True or False)
    :param disturbance_type: "zero" or "static" or "offset"
    :param dt_plant: Discretization time-step of plant
    :param dt_controller: Sampling rate of controller
    :param controller_params: Tuning parameters for controller
    For full-state feedback, Values of k1, k2, gamma_B, gamma_F, gamma_M of MRAC
    For output feedback, Values of a0 (zero of observer polynomial) and gamma
    :param init_estimate: Initial estimate of plant parameters, for controller initialization
    :return: results (dict)
    """
    assert np.isclose(dt_controller % dt_plant, 0, atol=0.0001)

    # Choice of system parameters as tuple (Me, B, A_sc, kf)
    params_plant = CASES[case]
    if not friction:
        params_plant[2] = 0.0

    plant = Linear_Motor_Continuous(params_plant, dt=dt_plant, cogging=cogging, disturbance_type=disturbance_type)
    ref_model = SecondOrderPlant_Continuous(num=[0, 0, REF_MODEL[0]**2],
                                            den=[1, 2*REF_MODEL[1]*REF_MODEL[0], REF_MODEL[0]**2],
                                            dt=dt_plant)
    ref_input = square_wave_continuous if ref_input_type == "square" else constant

    if controller_type == "adaptive":
        Controller = Controller_AC
    elif controller_type == "deterministic_robust":
        Controller = Controller_DRC
    elif controller_type == "deterministic_adaptive_robust":
        Controller = Controller_DARC
    else:
        raise ValueError("Wrong controller type!")

    init_estimate = INIT_ESTIMATE if not init_estimate else init_estimate
    controller = Controller(init_estimate=init_estimate, ref_model=ref_model,
                            controller_params=controller_params)

    time = 0.0
    measurement_history = [0.0]
    controller_history = [0.0]
    reference_inputs = [ref_input(0.0)]
    reference_outputs = [0.0]
    times = [0.0]

    while time < total_time:
        control_input = controller.get_control_input()
        plant.iterate(duration=dt_controller, control_input=control_input)
        ref_model.iterate(duration=dt_controller, control_input=ref_input(time))
        controller.iterate(duration=dt_controller, inputs=[plant.y, plant.y_dot])

        measurement_history.append(plant.y)
        controller_history.append(control_input)

        reference_inputs.append(ref_input(time))
        reference_outputs.append(ref_model.y)
        time += dt_controller
        times.append(time)

    mean_disturbance = 0.0
    if disturbance_type in ["static", "offset"]:
        mean_disturbance = 1.0

    true_params = params_plant[0:3] + params_plant[4:6] + [mean_disturbance]

    history = {"Measurements": measurement_history, "Parameters": controller.parameter_history,
               "ControlInputs": controller_history, "Method": controller_type + "_control",
               "TrueParameters": true_params, "ReferenceOutputs": reference_outputs,
               "ReferenceInputs": reference_inputs, "Time": times
               }

    return history


def constant(_, amplitude=0.2):
    """
    Used as a constant input for parameter tuning of controller
    """
    return amplitude
