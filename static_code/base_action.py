import math, copy
import numpy as np
from static_code.util_control_fun import *

def limit(value, min_value, max_value):
    return min(max(value, min_value), max_value)

def latlon_to_xyz(lat, lon, alt):
    from copy import deepcopy
    Latitude = deepcopy(lat)
    Longitude = deepcopy(lon)
    R = 637100
    L = 2 * math.pi * R
    Lat_l = L * np.cos(Latitude * math.pi/180)  # 当前纬度地球周长，度数转化为弧度
    Lng_l = 40030173  # 当前经度地球周长
    Lat_C = Lat_l / 360
    Lng_C = Lng_l / 360
    lat = Longitude * Lat_C#Latitude
    lon = Latitude * Lng_C#Longitude
    alt *= 0.3048
    return lat, lon, alt

class SimpleAction:
    
    def __init__(self, obs_dict, aircraft):
        self.aircraft = aircraft
        self.update(obs_dict) 
    
    def update(self, obs_dict):
        if obs_dict is None:
            return None
        self.obs_dict = obs_dict
        self.cur_alt = obs_dict['position_h_sl_ft']
        self.cur_pitch = obs_dict['attitude_pitch_rad']
        self.cur_roll = obs_dict['attitude_roll_rad']
        self.cur_yaw = obs_dict['attitude_psi_deg']
        self.cur_speed = obs_dict['velocities_mach']
        self.cur_v_down = obs_dict['velocities_v_down_fps']
        self.cur_long = obs_dict['position_long_gc_deg']
        self.cur_lat = obs_dict['position_lat_geod_deg']
        self.cur_x, self.cur_y, _ = latlon_to_xyz(self.cur_lat, self.cur_long, self.cur_alt)
    
    # 在某个高度，水平直线飞行
    def level_straight_fly(self, target_alt=5000, tolerance=4000):
        '''
            return: action
        '''
        action = {'aileron': 0, 'elevator': 0 , 'rudder': 0, 'throttle': 1}
        alt_err = target_alt - self.cur_alt
        
        if abs(alt_err) < tolerance:
            pass
        else:
            action['elevator'] = remap_curve_02(alt_err)
        
        # if self.cur_pitch >= 0.15 or self.cur_speed <= 0.9:
        #     action['throttle'] = 1.5
        # if self.cur_pitch <= -0.15 and alt_err < 0:
        #     action['throttle'] = 0.5
        
        action['aileron'] = 0 #self.make_roll(target_roll=0, tolerance=0)
        
        if self.cur_roll > math.pi/2 or self.cur_roll < -math.pi/2:
            action['elevator'] = -action['elevator']
        
        # print('lon: ' + str(self.cur_long) + ' lat: ' + str(self.cur_lat) + ' alt: ' + str(self.cur_alt) + ' roll: ' + str(self.cur_roll) + ' elevator: ' + str(action['elevator']) + ' aileron: ' + str(action['aileron']) + ' throttle: ' + str(action['throttle']))
        
        return action
    
    # 爬升到某个高度
    def climb(self, target_alt=10_000, tolerance=5):
        '''
            return: action['elevator']
        '''
        if self.cur_pitch >= 0.15 or self.cur_speed <= 0.2:
            elevator = 0.0
        else:
            elevator = -0.5
        if (target_alt - self.cur_alt) < tolerance:
            if self.cur_pitch >= 0.15:
                elevator = 0.1
            else:
                elevator = 0.0
        if self.cur_roll > math.pi/2 or self.cur_roll < -math.pi/2:
            elevator = -elevator
        return elevator
    
    # 下落到某个高度
    def drop(self, target_alt=10_000, tolerance=5):
        '''
            return: action['elevator']
        '''
        if self.cur_pitch <= -0.15:
            elevator = 0.0
        else:
            elevator = 0.3
        if (self.cur_alt - target_alt) < tolerance:
            if self.cur_pitch <= -0.15:
                elevator = -0.1
            else:
                elevator = 0.0
        if self.cur_roll > math.pi/2 or self.cur_roll < -math.pi/2:
            elevator = -elevator
        return elevator  
    
    # 翻滚到指定角度
    def make_roll(self, target_roll=math.pi/2, tolerance=0.1):
        '''
            return: action['aileron']
        '''
        roll_err = target_roll - self.cur_roll
        # print(roll_err)
        if abs(roll_err) < math.pi * tolerance / 180:
            return 0.0
        else:
            # return roll_err  / math.pi / 3
            return remap_curve_01(roll_err)
    
    # 方向舵转弯
    def make_yaw(self, target_yaw=0, tolerance=0.1):
        '''
            return: action['rudder']
        '''
        yaw_err = target_yaw - self.cur_yaw
        if abs(yaw_err) < math.pi * tolerance / 180:
            return 0.0
        else:
            return remap_curve_01(yaw_err)
    
    # 转弯
    def make_turn(self, target_yaw=180, tolerance=0.5):#2.5
        '''
            return: action['aileron']
        '''
        if self.aircraft == 'f16':
            yaw_err = target_yaw - self.cur_yaw
            if abs(yaw_err) < tolerance:
                return 0.0
            else:
                if abs(yaw_err) > 180:
                    if yaw_err > 0:
                        yaw_err = yaw_err - 360
                    else:
                        yaw_err = 360 + yaw_err
                # yaw_err += 10 if yaw_err > 0 else -10
                target_roll = limit(yaw_err / 180 * math.pi, -math.pi / 3, math.pi / 3)
                return self.make_roll(target_roll=target_roll)
        else:    
            yaw_err = target_yaw - self.cur_yaw
            if abs(yaw_err) < tolerance:
                return 0.0
            else:
                if abs(yaw_err) > 180:
                    if yaw_err > 0:
                        yaw_err = yaw_err - 360
                    else:
                        yaw_err = 360 + yaw_err
                yaw_err += 10 if yaw_err > 0 else -10
                target_roll = limit(yaw_err / 180 * math.pi, -math.pi / 3, math.pi / 3)
                return self.make_roll(target_roll=target_roll)
    
    # TODO: 开火，预留接口
    def fire(self, target_enemy):
        pass