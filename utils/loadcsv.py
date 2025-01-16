import os
import numpy as np
import math
import yaml

# def _fetch_motion_files(motion_file):
#     ext = os.path.splitext(motion_file)[1]
#     if (ext == ".yaml"):
#         dir_name = os.path.dirname(motion_file)
#         motion_files = []

#         with open(motion_file, 'r') as f:
#             motion_config = yaml.load(f, Loader=yaml.SafeLoader)

#         motion_list = motion_config['motions']
#         file_list = []
#         for motion_entry in motion_list:
#             curr_file = motion_entry['file']
            
#             file_list.append(curr_file.split('.')[0])

#             curr_file = os.path.join(dir_name, curr_file)
#             motion_files.append(curr_file)
#     else:
#         motion_files = [motion_file]
#     return motion_files, file_list

# def load_master_data(motion_file, master_config):
#     motion_files, file_list  = _fetch_motion_files(motion_file)
#     latent_obs_dict = {}
#     for f in range(len(motion_files)):
#         if file_list[f] in master_config:
#             curr_file = motion_files[f]
#             data = np.genfromtxt(curr_file, delimiter=',')[1:, 1:]
#             latent_obs_dict[file_list[f]] = data
#     return latent_obs_dict

def load_master_data(motion_file, master_config):
    paths = os.walk(motion_file)
    latent_obs_dict = {}
    for path, dir_lst, file_lst in paths:
        for file_name in file_lst:
            # print(str(master_config[file_name.split('_')[0]]) + str(file_name.split('_')[1].split('.')[0]).rjust(3, '0'))
            # print(os.path.join(path, file_name))
            latent_str = str(master_config[file_name.split('_')[0]]) + str(file_name.split('_')[1].split('.')[0]).rjust(3, '0')
            data = np.genfromtxt(os.path.join(path, file_name), delimiter=',', encoding='UTF-8')[1:, 3:9]
            latent_obs_dict[int(latent_str)] = data
    return latent_obs_dict
            

def trans_from_csv_to_mrad(org_obs):
    from copy import deepcopy
    org_obs = np.array(org_obs)
    Latitude = deepcopy(org_obs[..., 1])
    Longitude = deepcopy(org_obs[..., 0])
    R = 6371000
    L = 2 * math.pi * R
    Lat_l = L * np.cos(Latitude * math.pi/180)  # 当前纬度地球周长，度数转化为弧度
    Lng_l = R  # 当前经度地球周长
    Lat_C = Lat_l / 360
    Lng_C = Lng_l / 360
    org_obs[..., 0] = Longitude * Lat_C#Latitude
    org_obs[..., 1] = Latitude * Lng_C#Longitude
    
    # org_obs = np.array(org_obs)
    org_obs[..., 3:6] = np.deg2rad(org_obs[..., [4,3,5]])
    org_obs[..., 3] = - org_obs[..., 3]
    org_obs[..., 4] = - org_obs[..., 4]
    return org_obs

def trans_from_zjenv_to_mrad(org_obs):
    from copy import deepcopy
    org_obs = np.array(org_obs)
    Latitude = deepcopy(org_obs[..., 0])
    Longitude = deepcopy(org_obs[..., 1])
    R = 6371000
    L = 2 * math.pi * R
    Lat_l = L * np.cos(Latitude * math.pi/180)  # 当前纬度地球周长，度数转化为弧度
    Lng_l = R  # 当前经度地球周长
    Lat_C = Lat_l / 360
    Lng_C = Lng_l / 360
    org_obs[..., 0] = Longitude * Lat_C#Latitude
    org_obs[..., 1] = Latitude * Lng_C#Longitude
    
    org_obs[..., 2] *= 0.3048
    if org_obs.shape[-1] > 3:
        org_obs[..., 3], org_obs[..., 4] = - org_obs[..., 3], - org_obs[..., 4]
        org_obs[..., 5] = np.deg2rad(org_obs[..., 5])
    return org_obs

def trans_from_zjenv_to_csv(org_obs):
    org_obs = np.array(org_obs)
    org_obs[:, [0, 1]] = org_obs[:, [1, 0]]
    org_obs[..., 2] *= 0.3048
    
    org_obs[:, [3, 4]] = np.rad2deg(org_obs[:, [4, 3]])
    return org_obs

def write_result_csv_data(org_obs, file_path):
    import csv
    if type(org_obs) is not np.ndarray:
        org_obs = np.array(org_obs)
    line_id = np.array([[i] for i in range(org_obs.shape[0])])
    with open(file_path, "w", newline = '') as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(['Time', 'Longitude', 'Latitude', 'Altitude', 'Roll (deg)', 'Pitch (deg)', 'Yaw (deg)'])
        writer.writerows(np.hstack((line_id, org_obs)))

# def write_result_csv_data_multi_sheets(org_obs, file_path, sheet_names):
#     import csv
#     from openpyxl import Workbook
#     workbook = Workbook()
#     for id, name in enumerate(sheet_names):
#         line_id = np.array([[i] for i in range(org_obs[id].shape[0])])
#         new_sheet = workbook.create_sheet(name)
#         new_sheet.append(['Time', 'Longitude', 'Latitude', 'Altitude', 'Roll (deg)', 'Pitch (deg)', 'Yaw (deg)'])



def read_result_csv_data(file_path):
    import csv
    import numpy as np
    reader = csv.reader(open(file_path))
    result_traj = []
    for id, line in enumerate(reader):
        # print(line)
        if id != 0:
            result_traj.append(line[1:])
    return trans_from_csv_to_mrad(np.array(result_traj, dtype=np.float32))