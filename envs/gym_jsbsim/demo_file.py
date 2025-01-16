import gym
import gym_jsbsim
from gym_jsbsim.zj_jsbsim import zj_jsbsim
# ENV_ID = 'JSBSim-TurnHeadingControl-Cessna172P-SHAPING.STANDARD-NoFG-v0'
from gym_jsbsim.tasks import Shaping, HeadingControlTask, BattleTask
from gym_jsbsim.aircraft import Aircraft, cessna172P
from typing import Type, Tuple, Dict
import argparse
import numpy as np

parser = argparse.ArgumentParser()
# basic setting of the environment
parser.add_argument('--env_name', default="Jsbsim", type=str, help="the name of the environment")
parser.add_argument('--scenario_name', default="3vs3", type=str, help="3vs1 or 3vs3")
parser.add_argument('--max_timesteps', default=10000000, type=int)
parser.add_argument('--episode_length', default=500, type=int)
parser.add_argument('--n_actions', default=4, type=int)
parser.add_argument('--experiment_name', default='training', type=str)
parser.add_argument('--user_name', default='eexuefengw', type=str)
parser.add_argument('--attack_angle', default=40, type=int)
parser.add_argument('--attack_distance', default=200, type=int)

# number of agents and opponents
parser.add_argument('--num_agents', default=3, type=int)
parser.add_argument('--num_opponents', default=3, type=int)
parser.add_argument('--agent_keys', default=['player1', 'player2', 'player3'], type=list)
parser.add_argument('--opponent_keys', default=['player4', 'player5', 'player6'], type=list)

args = parser.parse_args()
aircrafts = ['f16', 'f16', 'f16', 'f16', 'f16', 'f16']
env = zj_jsbsim(args, BattleTask, aircrafts)
# env = gym.make('JSBSim-TurnHeadingControlTask-Cessna172P-Shaping.STANDARD-NoFG')

def _get_plane_info():
    plane_info_dict = {}
    # for key in self.agent_keys + self.opponent_keys:
    #     player_plane[key] = self.player_plane_dict[key]

    plane_info_dict['player1'] = {'initial_altitude_ft': 10000, 'initial_longitude': 38, 'initial_latitude': 41,
                                  'initial_heading_degree': 0, 'initial_velocity': 1000}
    plane_info_dict['player2'] = {'initial_altitude_ft': 10000, 'initial_longitude': 38.2, 'initial_latitude': 41,
                                  'initial_heading_degree': 0, 'initial_velocity': 1000}
    plane_info_dict['player3'] = {'initial_altitude_ft': 10000, 'initial_longitude': 38.1, 'initial_latitude': 41,
                                  'initial_heading_degree': 0, 'initial_velocity': 1000}
    plane_info_dict['player4'] = {'initial_altitude_ft': 10000, 'initial_longitude': 38.1, 'initial_latitude': 41.2,
                                  'initial_heading_degree': 180, 'initial_velocity': 1000}
    plane_info_dict['player5'] = {'initial_altitude_ft': 10000, 'initial_longitude': 38.2, 'initial_latitude': 41.2,
                                  'initial_heading_degree': 180, 'initial_velocity': 1000}
    plane_info_dict['player6'] = {'initial_altitude_ft': 10000, 'initial_longitude': 38.3, 'initial_latitude': 41.2,
                                  'initial_heading_degree': 180, 'initial_velocity': 1000}

    return plane_info_dict

initial_condition = _get_plane_info()

# env = zj_jsbsim()
state = env.reset(initial_condition)
for step in range(100):
    action = {}
    for i in range(6):
        action['player' + str(i + 1)] = {'aileron': 0, 'elevator': 0, 'rudder':0, 'throttle': 1}
    # actions = np.array([
    #     [0,0,0,1],
    #     [0,0,0,0],
    #     [0,0,0,0],
    #     [0,0,0,0],
    #     [0,0,0,0],
    #     [0,0,0,0],
    # ])
    state = env.step(action)

# print('state is {}', state[0])
print('hi')
# state, reward, done, info = env.step(action)