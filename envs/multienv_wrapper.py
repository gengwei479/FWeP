from datetime import datetime
import numpy as np
from envs.gym_jsbsim.zj_jsbsim import zj_jsbsim
from envs.gym_jsbsim.tasks import BattleTask, Shaping
from visual.plot_utils import mainWin
from static_code.rule_action import RuleAction
from utils.loadcsv import trans_from_zjenv_to_mrad, trans_from_zjenv_to_csv, write_result_csv_data

class jsbsimSingleMULENV(zj_jsbsim):
    def __init__(self, args, contain_rule=False, isConstistShow=False, isEndShow=False, auxiliary_line=None, csv_path=None, allow_flightgear_output=True):
        super().__init__(args, BattleTask, args.aircraft, agent_interaction=5, shaping=Shaping.STANDARD, allow_flightgear_output=allow_flightgear_output)
        self.initial_condition = {}
        self.agent_keys = args.agent_keys

        self.initial_condition[args.agent_keys[0]] = {
            'position_lat_geod_deg': 41, 'position_long_gc_deg': 38, 'position_h_sl_ft': 50000, 'initial_heading_degree': 0, 
            'velocities_v_east_fps': 0, 'velocities_v_north_fps': 1000, 'velocities_v_down_fps': 0, 'velocities_p_rad_sec': 0, 'velocities_q_rad_sec': 0, 'velocities_r_rad_sec': 0
        }
        self.initial_condition[args.agent_keys[1]] = {#41 38.0125
            'position_lat_geod_deg': 40.54, 'position_long_gc_deg': 38.23, 'position_h_sl_ft': 50000, 'initial_heading_degree': 0, 
            'velocities_v_east_fps': 0, 'velocities_v_north_fps': 1000, 'velocities_v_down_fps': 0, 'velocities_p_rad_sec': 0, 'velocities_q_rad_sec': 0, 'velocities_r_rad_sec': 0
        }#Giresun
        self.initial_condition[args.agent_keys[2]] = {#41 37.9875
            'position_lat_geod_deg': 40.54, 'position_long_gc_deg': 38.231, 'position_h_sl_ft': 50000, 'initial_heading_degree': 0, 
            'velocities_v_east_fps': 0, 'velocities_v_north_fps': 1000, 'velocities_v_down_fps': 0, 'velocities_p_rad_sec': 0, 'velocities_q_rad_sec': 0, 'velocities_r_rad_sec': 0
        }#Giresun

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
        
        self.obs_traj = {}
        self.rule_act = {}
        for agent_id, agent_name in enumerate(self.agent_keys):
            self.obs_traj[agent_name] = []
            self.rule_act[agent_name] = None
        self._max_episode_steps = 3000
        
    def reset(self, initial_condition = None):
        if initial_condition is None:
            obs = super().reset(self.initial_condition)
        else:
            obs = super().reset(initial_condition)
        
        for agent_id, agent_name in enumerate(self.agent_keys):
            self.obs_traj[agent_name].clear()
            self.obs_traj[agent_name].append(list(obs[agent_name].values())[0 : self.args.obs_dim])
        
            # if self.contain_rule:
            self.rule_act[agent_name] = RuleAction(obs[agent_name], aircraft = self.args.aircraft_type)
            self.rule_act[agent_name].set_original_info()
            self.rule_act[agent_name].set_start_step(0)
        
        return obs
    
    def step(self, action):
        
        if self.contain_rule:
            for agent_id, agent_name in enumerate(self.agent_keys):
                if self.rule_act[agent_name].cur_alt > self.altitude_steps[-1] or self.rule_act[agent_name].cur_alt < self.altitude_steps[1]:
                    mid_alt = (self.altitude_steps[-1] + self.altitude_steps[1]) / 2
                    action = {}
                    action[agent_name] = self.rule_act[agent_name].level_straight_fly(target_alt=mid_alt, tolerance=abs(self.altitude_steps[-1] - mid_alt))
        
        # if isinstance(action, list) or isinstance(action, np.ndarray):
        #     action = {'player1': dict(zip(['aileron', 'elevator', 'rudder', 'throttle'], action))}
        
        obs = super().step(action)

        for agent_id, agent_name in enumerate(self.agent_keys):
            self.rule_act[agent_name].update(obs[agent_name])
            self.obs_traj[agent_name].append(list(obs[agent_name].values())[0 : self.args.obs_dim])
        if self.isConstistShow:
            self.fig.draw(np.array(trans_from_zjenv_to_mrad(self.obs_traj['player1'])))
        return obs, action
    
    def complete(self):
        if self.csv_path is not None:
            current_datetime = datetime.now()
            formatted_date = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
            write_result_csv_data(trans_from_zjenv_to_csv(self.obs_traj), self.csv_path + formatted_date + '.csv')
        
        if self.isEndShow:
            self.fig.consist_update = False
            self.fig.draw(np.array(trans_from_zjenv_to_mrad(self.obs_traj['player1'])))

class MultiAgentFormation(jsbsimSingleMULENV):
    def __init__(self, args, contain_rule=False, isConstistShow=False, isEndShow=False, auxiliary_line=None, csv_path=None, destination=[41.17, 36.20, 50000], allow_flightgear_output=True, dash_stop = False):
        args.aircraft = [args.aircraft_type, args.aircraft_type, args.aircraft_type]
        args.num_agents = 3
        args.agent_keys = ['player1', 'player2', 'player3']
        super().__init__(args, contain_rule, isConstistShow, isEndShow, auxiliary_line, csv_path, allow_flightgear_output)

        self.leader_id = 0
        self.args = args
        self.env_name = "formation3"
        self.org_destination = trans_from_zjenv_to_mrad(destination)
        self.destination = self.org_destination
        self.step_num = 0
        self.dash_stop = dash_stop
        self.dest_name = "Samsun"

    def _rew_calculate(self, cur_obs):
        reward = [500 for _ in range(self.args.num_agents)]
        cur_pos = [trans_from_zjenv_to_mrad(list(cur_obs[name].values()))[:3] for name in self.args.agent_keys]

        for agent_id, agent_name in enumerate(self.agent_keys):
            if agent_id == self.leader_id:
                reward[agent_id] -= np.linalg.norm(self.destination[:2] - cur_pos[agent_id][:2]) / 1000
            else:
                rel_dist = np.linalg.norm(cur_pos[self.leader_id][:2] - cur_pos[agent_id][:2])
                if rel_dist < 50:
                    reward[agent_id] -= 200
                elif rel_dist > 100:
                    reward[agent_id] -= np.linalg.norm(cur_pos[self.leader_id][:2] - cur_pos[agent_id][:2]) / 1000
        
            if cur_pos[agent_id][-1] < 50:
                dash = -100
            else:
                dash = 0
            reward[agent_id] += dash
        
        return reward #sum(reward)

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
            if self.step_num >= self._max_episode_steps or trans_from_zjenv_to_mrad(list(obs[self.args.agent_keys[self.leader_id]].values()))[2] < 10:
                done = True
        else:
            if self.step_num >= self._max_episode_steps:
                done = True
        return obs, reward, done, action
    
    def complete(self):
        return super().complete()