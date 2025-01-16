# from environment import JsbSimEnv
import gym
import numpy as np
import random
from .tasks import Shaping, HeadingControlTask, BattleTask
from .simulation import Simulation
from .visualiser import FigureVisualiser, FlightGearVisualiser
from .aircraft import Aircraft, cessna172P
from typing import Type, Tuple, Dict

def dict_to_array(d):
  arr = []
  for key, value in d.items():
    if isinstance(value, dict):
      arr.append(dict_to_array(value))
    else:
      arr.append(value)
  return arr

def return_2_dict(keylist):
    sub_dict = {}
    keys = ['position_lat_geod_deg', 'position_long_gc_deg', 'position_h_sl_ft', 'attitude_pitch_rad',
            'attitude_roll_rad', 'attitude_psi_deg', 'velocities_mach', 'velocities_v_north_fps',
            'velocities_v_east_fps', 'velocities_v_down_fps', 'velocities_y', 'velocities_z',
            'velocities_p_rad_sec', 'velocities_q_rad_sec', 'velocities_r_rad_sec', 'last_aileron_left',
            'last_aileron_right', 'last_elevator', 'last_rudder']

    keylist = [dict(zip(keys, keylist[i])) for i in range(len(keylist))]

    for i in range(len(keylist)):
        sub_dict['player'+str(i+1)] = keylist[i]
    return sub_dict

class zj_jsbsim(gym.Env):
    """
    A class wrapping the JSBSim flight dynamics module (FDM) for simulating
    aircraft as an RL environment conforming to the OpenAI Gym Env
    interface.

    This version is set for multi-agent tasks

    ATTRIBUTION: this class implements the OpenAI Gym Env API. Method
    docstrings have been adapted or copied from the OpenAI Gym source code.
    """

    JSBSIM_DT_HZ: int = 60  # JSBSim integration frequency
    metadata = {'render.modes': ['human']}

    def __init__(self, args, task_type, aircraft, agent_interaction=5, shaping=Shaping.STANDARD, allow_flightgear_output = True):
        """
        Constructor. Inits some internal state, but JsbSimEnv.reset() must be
        called first before interacting with environment.

        :param args: arguments
        :param task_type: the Task subclass for the task agent is to perform
        :param aircraft: the JSBSim aircrafts to be used
        :param agent_interaction_freq: int, how many times per second the agent
            should interact with environment.
        :param shaping: a HeadingControlTask.Shaping enum, what type of agent_reward
            shaping to use (see HeadingControlTask for options)
        """
        if agent_interaction > self.JSBSIM_DT_HZ:
            raise ValueError('agent interaction frequency must be less than '
                             'or equal to JSBSim integration frequency of '
                             f'{self.JSBSIM_DT_HZ} Hz.')
        self.args = args
        self.num_agents = args.num_agents
        self.num_opponents = args.num_opponents
        self.num_total = self.num_opponents + self.num_agents
        self.task = task_type
        self.sims = None
        self.sim_steps_per_agent_step = int(self.JSBSIM_DT_HZ // agent_interaction)
        self.aircrafts = aircraft
        # self.tasks = [self.task(shaping, agent_interaction, aircraft[i]) for i in range(self.num_total)]

        self.tasks = [self.task(agent_interaction, aircraft[i]) for i in range(self.num_total)]
        # set Space objects
        self.observation_space: gym.spaces.Box = self.tasks[0].get_state_space()
        self.action_space: gym.spaces.Box = self.tasks[0].get_action_space()
        # set visualisation objects
        self.figure_visualiser = None
        self.flightgear_visualiser = None
        self.step_delay = None
        self.allow_flightgear_output = allow_flightgear_output

    def step(self, actions):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).

        :param action: the agent's action, with same length as action variables.
        :return:
            state: agent's observation of the current environment
            reward: amount of reward returned after previous action
            done: whether the episode has ended, in which case further step() calls are undefined
            info: auxiliary information, e.g. full reward shaping data
        """
        # if not (actions.shape == self.action_space.shape):
        #     raise ValueError('mismatch between action and action space size')
        actions = list(actions.values())
        state = [self.tasks[i].task_step(self.sims[i], list(actions[i].values()), self.sim_steps_per_agent_step) for i in range(self.num_total)]
        return return_2_dict(state)

    def init_condition(self):
        pass

    def obs_wrapper(self, obs):
        pass

    def _get_plane_info(self):
        player_planes = []
        # for key in self.agent_keys + self.opponent_keys:
        #     player_plane[key] = self.player_plane_dict[key]
        self.plane_info_dict = {}
        player_planes.append({'altitude': 10000, 'longitude': 38, 'latitude': 41, 'heading': 0, 'velocity': 1})
        player_planes.append({'altitude': 10000, 'longitude': 38.1, 'latitude': 41, 'heading': 0, 'velocity': 1})
        player_planes.append({'altitude': 10000, 'longitude': 38.2, 'latitude': 41, 'heading': 0, 'velocity': 1})
        player_planes.append({'altitude': 10000, 'longitude': 40, 'latitude': 41, 'heading': 180, 'velocity': 1})
        player_planes.append({'altitude': 10000, 'longitude': 40.1, 'latitude': 41, 'heading': 180, 'velocity': 1})
        player_planes.append({'altitude': 10000, 'longitude': 40.2, 'latitude': 41, 'heading': 180, 'velocity': 1})

        return player_planes

    def reset(self, initial_condition):
        """
        Resets the state of the environment and returns an initial observation.

        :return: array, the initial observation of the space.
        """

        # initial_conditions = self._get_plane_info()
        init_conditions = [self.tasks[i]._get_initial_conditions(initial_condition['player'+str(i+1)]) for i in range(self.num_total)]

        self.sims = [self._init_new_sim(self.JSBSIM_DT_HZ, self.aircrafts[i], init_conditions[i]) for i in range(self.num_total)]
        
        if self.allow_flightgear_output:
            for sim in self.sims:
                sim.enable_flightgear_output()
        else:
            for sim in self.sims:
                sim.disable_flightgear_output()

        state = [self.tasks[i].observe_first_state(self.sims[i]) for i in range(self.num_total)]

        # if self.flightgear_visualiser:
        #     self.flightgear_visualiser.configure_simulation_output(self.sims)
        obs_dict = return_2_dict(state)
        return obs_dict

    def _init_new_sim(self, dt, aircraft, initial_conditions):
        return Simulation(sim_frequency_hz=dt,
                          aircraft=aircraft,
                          init_conditions=initial_conditions)

    def close(self):
        """ Cleans up this environment's objects

        Environments automatically close() when garbage collected or when the
        program exits.
        """
        if self.sims:
            [self.sims[i].close() for i in range(self.num_total)]

    def seed(self, seed = None):
        """
        Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        gym.logger.warn("Could not seed environment %s", self)
        return

class zj_jsbsim_v2(gym.Env):
    """
    A class wrapping the JSBSim flight dynamics module (FDM) for simulating
    aircraft as an RL environment conforming to the OpenAI Gym Env
    interface.

    This version is set for multi-agent tasks

    ATTRIBUTION: this class implements the OpenAI Gym Env API. Method
    docstrings have been adapted or copied from the OpenAI Gym source code.
    """

    JSBSIM_DT_HZ: int = 60  # JSBSim integration frequency
    metadata = {'render.modes': ['human']}

    def __init__(self, args, task_type, aircraft, agent_interaction=5, shaping=Shaping.STANDARD):
        """
        Constructor. Inits some internal state, but JsbSimEnv.reset() must be
        called first before interacting with environment.

        :param args: arguments
        :param task_type: the Task subclass for the task agent is to perform
        :param aircraft: the JSBSim aircrafts to be used
        :param agent_interaction_freq: int, how many times per second the agent
            should interact with environment.
        :param shaping: a HeadingControlTask.Shaping enum, what type of agent_reward
            shaping to use (see HeadingControlTask for options)
        """
        if agent_interaction > self.JSBSIM_DT_HZ:
            raise ValueError('agent interaction frequency must be less than '
                             'or equal to JSBSim integration frequency of '
                             f'{self.JSBSIM_DT_HZ} Hz.')
        self.args = args
        self.num_agents = args.num_agents
        self.num_opponents = args.num_opponents
        self.num_total = self.num_opponents + self.num_agents
        self.task = task_type
        self.sims = None
        self.sim_steps_per_agent_step = int(self.JSBSIM_DT_HZ // agent_interaction)
        self.aircrafts = aircraft
        # self.tasks = [self.task(shaping, agent_interaction, aircraft[i]) for i in range(self.num_total)]

        self.tasks = [self.task(agent_interaction, aircraft[i]) for i in range(self.num_total)]
        # set Space objects
        self.observation_space: gym.spaces.Box = self.tasks[0].get_state_space()
        self.action_space: gym.spaces.Box = self.tasks[0].get_action_space()
        # set visualisation objects
        self.figure_visualiser = None
        self.flightgear_visualiser = None
        self.step_delay = None

    def step(self, actions):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).

        :param action: the agent's action, with same length as action variables.
        :return:
            state: agent's observation of the current environment
            reward: amount of reward returned after previous action
            done: whether the episode has ended, in which case further step() calls are undefined
            info: auxiliary information, e.g. full reward shaping data
        """
        # if not (actions.shape == self.action_space.shape):
        #     raise ValueError('mismatch between action and action space size')
        actions = list(actions.values())
        state = [self.tasks[i].task_step(self.sims[i], list(actions[i].values()), self.sim_steps_per_agent_step) for i in range(self.num_total)]
        return return_2_dict(state)

    def init_condition(self):
        pass

    def obs_wrapper(self, obs):
        pass

    def _get_plane_info(self):
        player_planes = []
        # for key in self.agent_keys + self.opponent_keys:
        #     player_plane[key] = self.player_plane_dict[key]
        self.plane_info_dict = {}
        player_planes.append({'altitude': 10000, 'longitude': 38, 'latitude': 41, 'heading': 0, 'velocity': 1})
        player_planes.append({'altitude': 10000, 'longitude': 38.1, 'latitude': 41, 'heading': 0, 'velocity': 1})
        player_planes.append({'altitude': 10000, 'longitude': 38.2, 'latitude': 41, 'heading': 0, 'velocity': 1})
        player_planes.append({'altitude': 10000, 'longitude': 40, 'latitude': 41, 'heading': 180, 'velocity': 1})
        player_planes.append({'altitude': 10000, 'longitude': 40.1, 'latitude': 41, 'heading': 180, 'velocity': 1})
        player_planes.append({'altitude': 10000, 'longitude': 40.2, 'latitude': 41, 'heading': 180, 'velocity': 1})

        return player_planes

    def reset(self, initial_condition):
        """
        Resets the state of the environment and returns an initial observation.

        :return: array, the initial observation of the space.
        """

        # initial_conditions = self._get_plane_info()
        init_conditions = [self.tasks[i]._get_initial_conditions(initial_condition['player'+str(i+1)]) for i in range(self.num_total)]

        self.sims = [self._init_new_sim(self.JSBSIM_DT_HZ, self.aircrafts[i], init_conditions[i]) for i in range(self.num_total)]

        state = [self.tasks[i].observe_first_state(self.sims[i]) for i in range(self.num_total)]

        # if self.flightgear_visualiser:
        #     self.flightgear_visualiser.configure_simulation_output(self.sims)
        obs_dict = return_2_dict(state)
        return obs_dict

    def _init_new_sim(self, dt, aircraft, initial_conditions):
        return Simulation(sim_frequency_hz=dt,
                          aircraft=aircraft,
                          init_conditions=initial_conditions)

    def close(self):
        """ Cleans up this environment's objects

        Environments automatically close() when garbage collected or when the
        program exits.
        """
        if self.sims:
            [self.sims[i].close() for i in range(self.num_total)]

    def seed(self, seed = None):
        """
        Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        gym.logger.warn("Could not seed environment %s", self)
        return