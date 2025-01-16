from datetime import datetime
import numpy as np
from envs.gym_jsbsim.zj_jsbsim import zj_jsbsim
from envs.gym_jsbsim.tasks import BattleTask, Shaping
from visual.plot_utils import mainWin
from static_code.rule_action import RuleAction
from utils.loadcsv import trans_from_zjenv_to_mrad, trans_from_zjenv_to_csv, write_result_csv_data

class jsbsimSingleENV(zj_jsbsim):
    def __init__(self, args, contain_rule=False, isConstistShow=False, isEndShow=False, auxiliary_line=None, csv_path=None, allow_flightgear_output=True):
        super().__init__(args, BattleTask, args.aircraft, agent_interaction=5, shaping=Shaping.STANDARD, allow_flightgear_output=allow_flightgear_output)
        self.initial_condition = {}
        self.initial_condition['player1'] = {
            'position_lat_geod_deg': 41, 'position_long_gc_deg': 38, 'position_h_sl_ft': 50000, 'initial_heading_degree': 0, 
            'velocities_v_east_fps': 0, 'velocities_v_north_fps': 1000, 'velocities_v_down_fps': 0, 'velocities_p_rad_sec': 0, 'velocities_q_rad_sec': 0, 'velocities_r_rad_sec': 0
        }
        self.args = args
        self.altitude_steps = args.altitude_steps
        self.contain_rule = contain_rule
        self.isConstistShow = isConstistShow
        self.isEndShow = isEndShow
        self.csv_path = csv_path
        if self.isConstistShow:
            self.fig = mainWin(consist_update=True, line_config=self.altitude_steps, auxiliary_line=auxiliary_line)
        else:
            self.fig = None
        self.obs_traj = []
        self.rule_act = None
        self._max_episode_steps = 3000
        
    def reset(self, initial_condition = None):
        if initial_condition is None:
            obs = super().reset(self.initial_condition)
        else:
            obs = super().reset(initial_condition)
        self.obs_traj.clear()
        self.obs_traj.append(list(obs['player1'].values())[0 : self.args.obs_dim])
        
        # if self.contain_rule:
        self.rule_act = RuleAction(obs['player1'], aircraft = self.args.aircraft_type)
        self.rule_act.set_original_info()
        self.rule_act.set_start_step(0)
        
        return obs
    
    def step(self, action):
        
        if self.contain_rule:
            if self.rule_act.cur_alt > self.altitude_steps[-1] or self.rule_act.cur_alt < self.altitude_steps[1]:
                mid_alt = (self.altitude_steps[-1] + self.altitude_steps[1]) / 2
                action = {}
                action['player1'] = self.rule_act.level_straight_fly(target_alt=mid_alt, tolerance=abs(self.altitude_steps[-1] - mid_alt))
        
        if isinstance(action, list) or isinstance(action, np.ndarray):
            action = {'player1': dict(zip(['aileron', 'elevator', 'rudder', 'throttle'], action))}
        
        obs = super().step(action)
        # if self.contain_rule:
        self.rule_act.update(obs['player1'])
        self.obs_traj.append(list(obs['player1'].values())[0 : self.args.obs_dim])
        if self.isConstistShow:
            self.fig.draw(np.array(trans_from_zjenv_to_mrad(self.obs_traj)))
        return obs, action
    
    def complete(self):
        if self.csv_path is not None:
            current_datetime = datetime.now()
            formatted_date = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
            write_result_csv_data(trans_from_zjenv_to_csv(self.obs_traj), self.csv_path + formatted_date + '.csv')
        
        if self.isEndShow:
            self.fig.consist_update = False
            self.fig.draw(np.array(trans_from_zjenv_to_mrad(self.obs_traj)))
            

#起点 Ordu （纬度）lat: 41  （经度）log: 38     高度：50000英尺 
#终点 Samsun （纬度）lat: 41.17  （经度）log: 36.20     高度：50000英尺 
class NavigateEnv(jsbsimSingleENV):
    def __init__(self, args, contain_rule=False, isConstistShow=False, isEndShow=False, auxiliary_line=None, csv_path=None, destination=[41.17, 36.20, 50000], allow_flightgear_output=True, dash_stop = False):
        super().__init__(args, contain_rule, isConstistShow, isEndShow, auxiliary_line, csv_path, allow_flightgear_output)
        # self.destination = [0, 0, 0] in meter
        self.env_name = "navigate"
        self.org_destination = trans_from_zjenv_to_mrad(destination)
        self.destination = self.org_destination
        self.step_num = 0
        self.dash_stop = dash_stop
        self.dest_name = "Samsun"
    
    def _rew_calculate(self, cur_obs):
        cur_pos = trans_from_zjenv_to_mrad(list(cur_obs['player1'].values()))[:3]
        distance = np.linalg.norm(self.destination[:2] - cur_pos[:2]) / 1000
        if cur_pos[-1] < 50:
            dash = -100
        else:
            dash = 0
        return -distance + dash + 500
    
    def reset(self, initial_condition = None):
        obs = super().reset(initial_condition)
        self.step_num = 0
        self.destination = self.org_destination
        return obs
        
    def step(self, action):
        obs, action = super().step(action)
        reward = self._rew_calculate(obs)
        self.step_num += 1
        
        done = False
        if self.dash_stop:
        # if np.linalg.norm(self.destination[:3] - trans_from_zjenv_to_mrad(list(obs['player1'].values()))[:3]) < 100 or self.step_num > self._max_episode_steps:
            if self.step_num >= self._max_episode_steps or trans_from_zjenv_to_mrad(list(obs['player1'].values()))[2] < 10:
                done = True
        else:
            if self.step_num >= self._max_episode_steps:
                done = True
        
        return obs, reward, done, action
    
    def complete(self):
        super().complete()

#起点 Ordu （纬度）lat: 41  （经度）log: 38     高度：50000英尺 
#终点0 Samsun （纬度）lat: 41.17  （经度）log: 36.20     高度：50000英尺 
#终点1 Rize （纬度）lat: 41.02  （经度）log: 40.31     高度：50000英尺 
class NavigateChangeEnv(NavigateEnv):
    def __init__(self, args, contain_rule=False, isConstistShow=False, isEndShow=False, auxiliary_line=None, csv_path=None, destination=[41.17, 36.2, 50000], allow_flightgear_output=True, dash_stop=False):
        super().__init__(args, contain_rule, isConstistShow, isEndShow, auxiliary_line, csv_path, destination, allow_flightgear_output, dash_stop)
        self.env_name = "nav_change"
        self.change_step = 1000
        self.dest_name = "Samsun"

    def _rew_calculate(self, cur_obs):
        return super()._rew_calculate(cur_obs)
    
    def reset(self, initial_condition=None):
        self.dest_name = "Samsun"
        return super().reset(initial_condition)
    
    def step(self, action):
        if self.step_num >= self.change_step:
            self.destination = trans_from_zjenv_to_mrad([40.55, 35.55, 50000])#[41.02, 40.31, 50000] [39.44, 39.29, 50000]
            self.dest_name = "Ladik"#Rize Erzincan
        return super().step(action)
    
    def complete(self):
        return super().complete()

class AvoidCollisionEnv(jsbsimSingleENV):
    def __init__(self, args, contain_rule=False, isConstistShow=False, isEndShow=False, auxiliary_line=None, csv_path=None, destination=[41.17, 36.20, 5000], collisionPoints=[{'pos': [41.13, 37, 5000], 'radis': 20000}], allow_flightgear_output=True, dash_stop = False):
        super().__init__(args, contain_rule, isConstistShow, isEndShow, auxiliary_line, csv_path, allow_flightgear_output)
        # destination = [0, 0, 0] in meter
        # collisionPoints = [{'pos': [0, 0, 0], 'radis': 10}, ...] in meter
        self.env_name = "nav_collision"
        self.destination = trans_from_zjenv_to_mrad(destination)
        self.collisionPoints = collisionPoints
        self.step_num = 0
        self.dash_stop = dash_stop
        self.dest_name = "Samsun. But you need to avoid the restricted area for at least {} meters. The location of the restricted area is {}".format(collisionPoints[0]['radis'], trans_from_zjenv_to_mrad(collisionPoints[0]['pos']))
    
    def _rew_calculate(self, cur_obs):
        reward = 0
        cur_pos = trans_from_zjenv_to_mrad(list(cur_obs['player1'].values()))[:3]
        reward -= np.linalg.norm(self.destination[:2] - cur_pos[:2]) / 1000
        
        if cur_pos[-1] < 50:
            dash = -100
        else:
            dash = 0
        reward += dash
        
        for col in self.collisionPoints:
            if np.linalg.norm(col['pos'][:3] - cur_pos) < col['radis']:
                reward -= (col['radis'] - np.linalg.norm(col['pos'][:2] - cur_pos[:2])) / 200
        return reward + 500
    
    def reset(self, initial_condition = None):
        obs = super().reset(initial_condition)
        self.step_num = 0
        return obs
        
    def step(self, action):
        obs, action = super().step(action)
        reward = self._rew_calculate(obs)
        self.step_num += 1
        
        done = False
        if self.dash_stop:
        # if np.linalg.norm(self.destination[:3] - trans_from_zjenv_to_mrad(list(obs['player1'].values()))[:3]) < 100 or self.step_num > self._max_episode_steps:
            if self.step_num >= self._max_episode_steps or trans_from_zjenv_to_mrad(list(obs['player1'].values()))[2] < 10:
                done = True
        else:
            if self.step_num >= self._max_episode_steps:
                done = True
        return obs, reward, done, action
    
    def complete(self):
        super().complete()