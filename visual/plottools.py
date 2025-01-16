import math
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def lower_bound(list, item):
    left = 0
    right = len(list)
    mid = int((left + right) / 2)
    while left < right:
        mid = int((left + right) / 2)
        if list[mid] >= item:
            right = mid
        else:
            left = mid + 1
    return left - 1

def rotate_trans(roll, pitch, yaw):
    roll_mat = [[1,0,0],[0,math.cos(roll),-math.sin(roll)],[0,math.sin(roll),math.cos(roll)]]
    pitch_mat = [[math.cos(pitch),0,math.sin(pitch)],[0,1,0],[-math.sin(pitch),0,math.cos(pitch)]]
    yaw_mat = [[math.cos(yaw),-math.sin(yaw),0],[math.sin(yaw),math.cos(yaw),0],[0,0,1]]
    return np.mat(roll_mat) * np.mat(pitch_mat) * np.mat(yaw_mat)

def model_data(o, rotation, size=(10, 10, 10)):
    # X = [[[1, 0, 0], [-1, 1, 0], [-1, -1, 0]],
    #      [[1, 0, 0], [-1, 0, 0], [-1.25, 0, 0.75]]]
    X = [[[0, 1, 0], [1, -1, 0], [-1, -1, 0]],
         [[0, 1, 0], [0, -1, 0], [0, -1.25, 0.75]]]
    
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] = (np.mat(X[i][j]) @ rotate_trans(rotation[0], rotation[1], rotation[2])).tolist()[0]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:, :, i] *= size[i]
    X += np.array(o)
    return X
 
 
def plotPlane(positions, rotation=None, sizes=None, colors=None, **kwargs):
    if not isinstance(colors, (list, np.ndarray)):
        colors = ["C0"] * len(positions)
    if not isinstance(sizes, (list, np.ndarray)):
        sizes = [(1, 1, 1)] * len(positions)
    g = []
    for p, s, c in zip(positions, sizes, colors):
       g.append(model_data(p, rotation, size=s))
    return Poly3DCollection(np.concatenate(g), facecolors=np.repeat(colors, 6), alpha=0.6, **kwargs)

def plotPoint(positions, ax, colors = None):
    for id, position in enumerate(positions):
        if colors is None:
            ax.scatter(position[0], position[1], position[2], c='black')
        else:
            ax.scatter(position[0], position[1], position[2], c=colors[id])

def plotCurve(positions, ax, line_config, colors):
    plot_res = {}
    tmp_flag = ''
    for item in positions:
        tmp_key = colors[lower_bound(line_config, item[2])]
        if tmp_flag != tmp_key:
            if not tmp_key in plot_res.keys():
                plot_res[tmp_key] = []
            plot_res[tmp_key].append([])
            tmp_flag = tmp_key

        plot_res[tmp_key][len(plot_res[tmp_key]) - 1].append(item)
            
    for key, value in plot_res.items():
        for positions in value:
            positions = np.array(positions)
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color=key)

def plotCurve01(positions, ax, color):
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color = color, linestyle = '--')