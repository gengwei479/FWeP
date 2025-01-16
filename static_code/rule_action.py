import math
import numpy as np

from static_code.base_action import SimpleAction


class RuleAction(SimpleAction):
    
    def __init__(self, obs_dict, aircraft = ''):
        super().__init__(obs_dict, aircraft)
        self.aircraft = aircraft
        self.flag = False
    
    def set_start_step(self, step):
        self.start_step = step
    
    def set_original_info(self):
        self.original_x = self.cur_x
        self.original_y = self.cur_y
        self.original_alt = self.cur_alt
        self.original_yaw = self.cur_yaw
    
    # 前往某个点所在的方向
    def goto_point(self, action, target_add_yaw=180, target_add_alt=1000):
        '''
            target_add_x: >0: north(yaw=0) <0: south(yaw=180)
            target_add_y: >0: east(yaw=90) <0: west(yaw=270)
            target_add_alt: >0: up, <0: down
        '''
        if self.aircraft == 'f16':
            if abs(target_add_alt) > 50:
                if target_add_alt > 0:
                    action['elevator'] = self.climb(target_alt=self.original_alt + target_add_alt)
                else:
                    action['elevator'] = self.drop(target_alt=self.original_alt + target_add_alt)
            action['aileron'] = self.make_turn(target_yaw=self.original_yaw + target_add_yaw)
            # action['rudder'] = self.make_turn(target_yaw=target_yaw)
        else:
            # action = dict(zip(['aileron', 'elevator', 'rudder', 'throttle'], [0, 0, 0, 1]))
            if abs(target_add_alt) > 50:
                if target_add_alt > 0:
                    action['elevator'] = self.climb(target_alt=self.original_alt + target_add_alt)
                else:
                    action['elevator'] = self.drop(target_alt=self.original_alt + target_add_alt)
            action['aileron'] = self.make_turn(target_yaw=self.original_yaw + target_add_yaw)
        return action

    def adjust_to_north(self, action, target_add_yaw = 0, target_add_alt=0):
        if abs(target_add_alt) > 50:
            if target_add_alt > 0:
                action['elevator'] = self.climb(target_alt=self.original_alt + target_add_alt)
            else:
                action['elevator'] = self.drop(target_alt=self.original_alt + target_add_alt)
        action['rudder'] = self.make_yaw(target_yaw=self.original_yaw + target_add_yaw)
        return action
    
    # 滚筒动作，用于躲避咬尾
    def barrel(self, action, step):
        if self.cur_alt < 1000:
            print('too low to barrel')
            return action, True
        if self.cur_speed < 0.5:
            print('too slow to barrel')
            return action, True
        
        if step - self.start_step <= 5:
            action['aileron'] = self.make_roll(target_roll=0, tolerance=2)  # 先滚平
        action['aileron'] = 0.15
        action['elevator'] = -0.4 if abs(self.cur_roll) < math.pi * 150 / 180 else -0.1
        action_complete = False if (step - self.start_step) < 70 else True
        return action#, action_complete
    
    # 破S动作，用于急速下落
    def split_s(self, action, step, target_alt=2000):
        if step - self.start_step <= 5:
            action['aileron'] = self.make_roll(target_roll=0, tolerance=2)  # 先滚平
        elif step - self.start_step <= 10:
            action['aileron'] = 0.65
            action['elevator'] = -0.1
        else:
            action['elevator'] = -1.5
            if abs(self.cur_pitch - math.pi/4) < math.pi * 5 / 180:
                action['elevator'] = -0.2
            if abs(self.cur_alt - target_alt) < 1000 or (step - self.start_step) > 60:
                if self.cur_alt > target_alt:
                    action['elevator'] = self.drop(target_alt=target_alt)
                else:
                    action['elevator'] = self.climb(target_alt=target_alt)
            if abs(self.cur_alt - target_alt) < 100:
                action = self.level_straight_fly(target_alt=target_alt)
        action_complete = False if (step - self.start_step) < 70 else True
        return action#, action_complete
    
    # 殷麦曼，上升倒转
    def immel(self, action, step, target_alt=15000):
        if (step - self.start_step) <=5:
            action['aileron'] = self.make_roll(target_roll=0, tolerance=2)  # 先滚平
        else:
            action['elevator'] = -2
        if (step - self.start_step) > 60:
            action['elevator'] = 0.0
            action['aileron'] = self.make_roll(target_roll=0, tolerance=2)  # 滚平
            if abs(self.cur_roll) < math.pi * 2 / 180:
                if self.cur_alt > target_alt:
                    action['elevator'] = self.drop(target_alt=target_alt)
                else:
                    action['elevator'] = self.climb(target_alt=target_alt)
                action = self.level_straight_fly(target_alt=target_alt)
        # action_complete = False if (step - self.start_step) < 70 else True
        return action#, action_complete
    
    # 筋斗
    def loop(self, action, step, target_alt=10000):
        if (step - self.start_step) > 100 and abs(self.cur_pitch) < math.pi * 2 / 180:
            self.flag = True
        if self.flag:
            if self.cur_alt > target_alt:
                action['elevator'] = self.drop(target_alt=target_alt)
            else:
                action['elevator'] = self.climb(target_alt=target_alt)
            return action
        if (step - self.start_step) <= 5:
            action['aileron'] = self.make_roll(target_roll=0, tolerance=2)  # 先滚平
        else:
            action['elevator'] = -1.0
        # action_complete = False if (step - self.start_step) < 70 else True
        return action#, action_complete
    
    # 死亡后迫降（退出战场）
    def crash(self, target_alt = 1000):
        """
        move to the lowest altitude directly if the airplane is dead; stay still if already located at the lowest altitude.

        Args:
            target_alt (int, optional): the lowest altitude. Defaults to 1000.
        
        Return:
            action (elevator)
        """
        if self.original_alt >= target_alt:
            action = self.drop(target_alt=target_alt, tolerance=200)
        else:
            action = self.drop(target_alt=self.original_alt, tolerance=200)
        
        return action
    
    # def idle(self, torlance = 0.01, init_action = [0, 0, 0, 1], is_stable = False):
    #     action = dict(zip(['aileron', 'elevator', 'rudder', 'throttle'], init_action))
    #     action['elevator'] = 0 if abs(self.cur_pitch) < torlance else 5 * self.cur_pitch / math.pi#15
    #     action['aileron'] = 0 if abs(self.cur_roll) < torlance else 5 * -self.cur_roll / math.pi#2
    #     return action, is_stable
    
    def idle(self, torlance = 0.01, init_action = [0, 0, 0, 1], is_stable = False):
        action = dict(zip(['aileron', 'elevator', 'rudder', 'throttle'], init_action))
        
        if not is_stable:
            action['elevator'] = - 0.01 * self.cur_v_down#-0.01
        else:
            action['elevator'] = 0 if abs(self.cur_pitch) < torlance else 5 * self.cur_pitch / math.pi
        action['aileron'] = 0 if abs(self.cur_roll) < torlance else 5 * -self.cur_roll / math.pi
        
        if abs(self.cur_pitch) < torlance and abs(self.cur_roll) < torlance and abs(self.cur_v_down) < 1:
            is_stable = True
        else:
            is_stable = False
        return action, is_stable