"""
Model Reference Adaptive Controllers
------------------------------------
Considers the linear motor plant given by the TF 1/(M_e s^2 + Bs),
with additional (unknown) Coulomb friction. Uses a MRAController to
track a second order reference nodel.
"""
import numpy as np
from copy import deepcopy
from control_library import Linear_Motor_Continuous, SecondOrderPlant_Continuous, CASES, square_wave_continuous,\
    column, StateSpacePlant

# These are arbitrary and not tuned!
INIT_ESTIMATE = {"full_state_feedback": (0.055, 0.225, 0.125),  # M_e, B and A_sc
                 "output_feedback": (1.0, 1.0, 1.0, 1.0, 1.0)}  # r'1, s0, s1, t0, t1
REF_MODEL = (15, 1)  # Nat_freq. and damping ratio


class Controller_FSF:
    def __init__(self, init_estimate, ref_model, controller_params):
        """
        :param init_estimate: Me, B and A_sc
        :param ref_model:
        :param controller_params: Values of k1, k2, gamma_B, gamma_F, gamma_M of MRAC
        """
        assert isinstance(ref_model, SecondOrderPlant_Continuous)
        assert len(controller_params) == 5
        assert len(init_estimate) == 3

        self.M_e, self.B, self.A_sc = init_estimate
        self.kf = 1000  # We assume that this is known!
        self.ref_model = ref_model
        self.k1, self.k2, self.gamma_B, self.gamma_F, self.gamma_M = controller_params

        self.parameter_history = [deepcopy([self.M_e, self.B, self.A_sc])]
        self.control_input = self.M_e*(self.ref_model.get_y_ddot(square_wave_continuous(0.0)))
        self.time = 0.0

    def iterate(self, duration, inputs):
        self.time += duration
        y, y_dot = inputs
        sat_kf_y = np.clip(self.kf * y_dot, -1, 1)
        ym, ym_dot = self.ref_model.y, self.ref_model.y_dot
        p = y_dot - ym_dot + self.k1*(y-ym)

        # Update parameters
        self.B -= self.gamma_B*y_dot*p * duration
        self.A_sc -= self.gamma_F*sat_kf_y*p * duration
        self.M_e -= self.gamma_M*(self.ref_model.y_ddot - self.k1*(y_dot-ym_dot))*p * duration

        self.control_input = self.B*y_dot + self.A_sc*sat_kf_y \
            + self.M_e*(self.ref_model.y_ddot - self.k1*(y_dot-ym_dot)) - self.k2*p

        self.parameter_history.append(deepcopy([self.M_e, self.B, self.A_sc]))

    def get_control_input(self):
        return self.control_input


class Controller_OF:
    def __init__(self, init_estimate, ref_model, controller_params):
        """
        :param init_estimate: Initial params of control law, [b_0, r'1, s0, s1, t0, t1]
        :param ref_model:
        :param controller_params: Values of a0 (zero of observer polynomial), gamma_b and gamma[0:5]
        """
        assert isinstance(ref_model, SecondOrderPlant_Continuous)
        assert len(controller_params) == 7
        assert len(init_estimate) == 6

        self.b_0 = init_estimate[0]
        self.params = column(init_estimate[1:6])  # [r'1 s0 s1 t0 t1]

        self.a0 = controller_params[0]
        self.gamma_b = controller_params[1]
        self.gamma = np.diag(controller_params[2:7])

        self.ref_model = ref_model
        self.filters = {"u": {}, "y": {}, "neg_uc": {}}
        for filters in self.filters.values():
            filters["P2"] = StateSpacePlant(A=[[-self.a0]],
                                            B=[[1]],
                                            dt=self.ref_model.dt)
            filters["P"] = StateSpacePlant(A=[[-self.ref_model.den[1], -self.ref_model.den[2]],
                                              [1, 0]],
                                           B=[[1], [0]],
                                           dt=self.ref_model.dt)
        self.reg_filtered = {"P1": column(np.zeros(1)),  # [u] / P1
                             "P2": column(np.zeros(5)),  # [u py y -puc -uc] / P2
                             "P": column(np.zeros(5))}  # [u py y -puc -uc] / (P1.P2)

        self.parameter_history = [list(deepcopy(self.params.T[0]))]
        self.parameter_history[-1].append(self.b_0)
        self.control_input = self.get_control_input()
        self.time = 0.0

    def iterate(self, duration, inputs):
        y = inputs[0]
        self.time += duration
        ym, uc = self.ref_model.y, self.ref_model.control_input
        error = y - ym

        param_error = -(float(self.reg_filtered["P1"][0]) + float(self.reg_filtered["P"].T @ self.params))
        augmented_error = error + self.b_0*param_error

        self.b_0 = self.b_0 - self.gamma_b*augmented_error*param_error*duration
        self.params = self.params + augmented_error*(self.gamma @ self.reg_filtered["P"])*duration*np.sign(self.b_0)
        self.parameter_history.append(list(deepcopy(self.params.T[0])))
        self.parameter_history[-1].append(self.b_0)

        self.filters["u"]["P2"].iterate(duration, control_input=[[self.control_input]])
        self.filters["y"]["P2"].iterate(duration, control_input=[[y]])
        self.filters["neg_uc"]["P2"].iterate(duration, control_input=([[-self.ref_model.control_input]]))

        self.filters["u"]["P"].iterate(duration, control_input=[self.filters["u"]["P2"].state[-1]])
        self.filters["y"]["P"].iterate(duration, control_input=[self.filters["y"]["P2"].state[-1]])
        self.filters["neg_uc"]["P"].iterate(duration, control_input=[self.filters["neg_uc"]["P2"].state[-1]])

        self.reg_filtered["P2"] = column([self.filters["u"]["P2"].state[-1],
                                          self.filters["y"]["P2"].derivative[-1],
                                          self.filters["y"]["P2"].state[-1],
                                          self.filters["neg_uc"]["P2"].derivative[-1],
                                          self.filters["neg_uc"]["P2"].state[-1]
                                          ])
        self.reg_filtered["P"] = column([self.filters["u"]["P"].state[-1],
                                         self.filters["y"]["P"].state[-2],
                                         self.filters["y"]["P"].state[-1],
                                         self.filters["neg_uc"]["P"].state[-2],
                                         self.filters["neg_uc"]["P"].state[-1]
                                         ])
        self.reg_filtered["P1"] = self.filters["u"]["P"].state[0] + self.a0*self.filters["u"]["P"].state[1]
        # import pdb; pdb.set_trace()

    def get_control_input(self):
        self.control_input = -float(self.params.T @ self.reg_filtered["P2"])
        return self.control_input


def simulate(total_time=2.0, case=1, controller_type="output_feedback", friction=False, dt_plant=None,
             dt_controller=None, controller_params: tuple = None, init_estimate: list = None):
    """
    Simulates linear motor with friction and MRAC controller
    :param total_time: Total time of simulation
    :param case: Choice of system parameters (see control_library.py
    :param controller_type: full_state_feedback or output_feedback
    :param friction: Simulate friction (True or False)
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

    plant = Linear_Motor_Continuous(params_plant, dt=dt_plant)
    ref_model = SecondOrderPlant_Continuous(num=[0, 0, REF_MODEL[0]**2],
                                            den=[1, 2*REF_MODEL[1]*REF_MODEL[0], REF_MODEL[0]**2],
                                            dt=dt_plant)

    if controller_type == "full_state_feedback":
        Controller = Controller_FSF
    elif controller_type == "output_feedback":
        Controller = Controller_OF
    else:
        raise ValueError("Invalid controller type!")

    init_estimate = INIT_ESTIMATE[controller_type] if not init_estimate else init_estimate
    controller = Controller(init_estimate=init_estimate, ref_model=ref_model,
                            controller_params=controller_params)

    time = 0.0
    measurement_history = [0.0]
    controller_history = [0.0]
    reference_inputs = [square_wave_continuous(0.0)]
    reference_outputs = [0.0]
    times = [0.0]

    while time < total_time:
        control_input = controller.get_control_input()
        plant.iterate(duration=dt_controller, control_input=control_input)
        ref_model.iterate(duration=dt_controller, control_input=square_wave_continuous(time))
        controller.iterate(duration=dt_controller, inputs=[plant.y, plant.y_dot])
        time += dt_controller

        measurement_history.append(plant.y)
        controller_history.append(control_input)

        reference_inputs.append(square_wave_continuous(time))
        reference_outputs.append(ref_model.y)
        times.append(time)

    history = {"Measurements": measurement_history, "Parameters": controller.parameter_history,
               "ControlInputs": controller_history, "Method": "MRAC with " + controller_type,
               "TrueParameters": params_plant[0:3] if controller_type == "full_state_feedback" else None,
               "ReferenceOutputs": reference_outputs, "ReferenceInputs": reference_inputs,
               "Time": times
               }

    return history


