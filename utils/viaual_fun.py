import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import  ConnectionPatch

def zone_and_linked(ax,axins,zone_left,zone_right,x,y,linked='bottom',
                    x_ratio=0.05,y_ratio=0.05):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
    xlim_right = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data)-(np.max(y_data)-np.min(y_data))*y_ratio
    ylim_top = np.max(y_data)+(np.max(y_data)-np.min(y_data))*y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left,xlim_right,xlim_right,xlim_left,xlim_left],
            [ylim_bottom,ylim_bottom,ylim_top,ylim_top,ylim_bottom],"black")

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_left,ylim_bottom)
        xyA_2, xyB_2 = (xlim_right,ylim_top), (xlim_right,ylim_bottom)
    elif  linked == 'top':
        xyA_1, xyB_1 = (xlim_left,ylim_bottom), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_right,ylim_top)
    elif  linked == 'left':
        xyA_1, xyB_1 = (xlim_right,ylim_top), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_left,ylim_bottom)
    elif  linked == 'right':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_right,ylim_top)
        xyA_2, xyB_2 = (xlim_left,ylim_bottom), (xlim_right,ylim_bottom)
        
    con = ConnectionPatch(xyA=xyA_1,xyB=xyB_1,coordsA="data",
                          coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2,xyB=xyB_2,coordsA="data",
                          coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)


def line_graphs_01(inputsX, inputsY, label, dir):
    pic_obj=plt.figure()#figsize=(4, 3), dpi=300
    fig = plt.subplot(111, facecolor = '#EBEBEB')
    for key, value in inputsY.items():
        x_data = []
        y_data = []
        for j in value:
            # assert len(inputsX) == len(j)
            inputsX = inputsX[:min(len(inputsX), len(j))]
            j = j[:min(len(inputsX), len(j))]
            x_data += inputsX
            if hasattr(j, 'tolist'):
                y_data += j.tolist()
            else:
                y_data += j
        data = pd.DataFrame({'x': x_data, 'y': y_data})
        sns.lineplot(data = data, x = 'x', y = 'y', label = key)
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    fig.spines['top'].set_visible(False)
    fig.spines['right'].set_visible(False)
    plt.grid(c='w')
    plt.legend()
    # plt.show()
    pic_obj.savefig(dir)

# Ghostscript
# conda install conda-forge::miktex
def line_multi_graphs_01(inputsX, inputsYs, label, dir, desc_info):
    # fig = plt.figure()
    # axs = fig.subplots(nrows=2, ncols=4)
    plt.rcParams['text.usetex']=True
    plt.rcParams['text.latex.preamble']=r'\makeatletter \newcommand*{\rom}[1]{\expandafter\@slowromancap\romannumeral #1@} \makeatother'
    # roman_nums = ['I', 'II', 'III']
    fontsize = 50
    nrows = 2
    ncols = 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(60, 27))
    for id, inputsY in enumerate(inputsYs):
        for key, value in inputsY.items():
            x_data = []
            y_data = []
            for j in value:
                inputsX = inputsX[:min(len(inputsX), len(j))]
                j = j[:min(len(inputsX), len(j))]
                x_data += inputsX
                if hasattr(j, 'tolist'):
                    y_data += j.tolist()
                else:
                    y_data += j
            data = pd.DataFrame({'x': x_data, 'y': y_data})
            sns.lineplot(data = data, x = 'x', y = 'y', label = key, ax=axs[int(id / ncols), id % ncols], legend=False)

        axs[int(id / ncols), id % ncols].set_title(r'(\rom{})'.format(id+1) + ' {} of {}'.format(desc_info[id]['task_name'], desc_info[id]['air_craft']), fontsize=fontsize)
        axs[int(id / ncols), id % ncols].set_xlabel(label[0], fontsize=fontsize)
        axs[int(id / ncols), id % ncols].set_ylabel(label[1], fontsize=fontsize)
        axs[int(id / ncols), id % ncols].tick_params(axis='both', labelsize=fontsize)
        axs[int(id / ncols), id % ncols].spines['top'].set_visible(False)
        axs[int(id / ncols), id % ncols].spines['bottom'].set_visible(False)
        axs[int(id / ncols), id % ncols].spines['left'].set_visible(False)
        axs[int(id / ncols), id % ncols].spines['right'].set_visible(False)
        axs[int(id / ncols), id % ncols].patch.set_facecolor('#EBEBEB')
        axs[int(id / ncols), id % ncols].grid(True, color = '#FFFFFF')
    
    fig.subplots_adjust(bottom=0.15, left=0.04, right=0.995, top=0.96, wspace=0.2, hspace=0.3)
    lines, labels = axs[-1, -1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'lower center', ncol=5, borderaxespad=0, handlelength=4, frameon = False, fontsize=fontsize)
    fig.savefig(dir)

def line_multi_graphs_01_c(inputsX, inputsYs, label, dir, desc_info, fontsize = 50, fig_size = (60, 27)):
    # fig = plt.figure()
    # axs = fig.subplots(nrows=2, ncols=4)
    plt.rcParams['text.usetex']=True
    plt.rcParams['text.latex.preamble']=r'\makeatletter \newcommand*{\rom}[1]{\expandafter\@slowromancap\romannumeral #1@} \makeatother'
    # roman_nums = ['I', 'II', 'III']
    fontsize = fontsize
    nrows = 1
    ncols = 2
    fig, axs = plt.subplots(nrows, ncols, figsize=fig_size)
    for id, inputsY in enumerate(inputsYs):
        for key, value in inputsY.items():
            x_data = []
            y_data = []
            for j in value:
                inputsX = inputsX[:min(len(inputsX), len(j))]
                j = j[:min(len(inputsX), len(j))]
                x_data += inputsX
                if hasattr(j, 'tolist'):
                    y_data += j.tolist()
                else:
                    y_data += j
            data = pd.DataFrame({'x': x_data, 'y': y_data})
            sns.lineplot(data = data, x = 'x', y = 'y', label = key, ax=axs[id], legend=False)

        axs[id].set_title(r'(\rom{})'.format(id+1) + ' {}'.format(desc_info[id]['air_craft']), fontsize=fontsize)
        axs[id].set_xlabel(label[0], fontsize=fontsize)
        axs[id].set_ylabel(label[1], fontsize=fontsize)
        axs[id].tick_params(axis='both', labelsize=fontsize)
        axs[id].spines['top'].set_visible(False)
        axs[id].spines['bottom'].set_visible(False)
        axs[id].spines['left'].set_visible(False)
        axs[id].spines['right'].set_visible(False)
        axs[id].patch.set_facecolor('#EBEBEB')
        axs[id].grid(True, color = '#FFFFFF')
    
    fig.subplots_adjust(bottom=0.15, left=0.04, right=0.995, top=0.96, wspace=0.2, hspace=0.3)
    lines, labels = axs[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'lower center', ncol=5, borderaxespad=0, handlelength=4, frameon = False, fontsize=fontsize)
    fig.savefig(dir)
    

def line_graphs_02(inputsX, inputsY, label, dir, y_label_list = None, x_start = 1400, x_end = 1499):
    fig, ax = plt.subplots(1,1)
    for key, value in inputsY.items():
        ax.plot(inputsX, value, label = key)
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    # fig.spines['top'].set_visible(False)
    # fig.spines['right'].set_visible(False)
    plt.grid(c='w')
    plt.legend()

    axins = ax.inset_axes((0.25, 0.45, 0.25, 0.5))
    for key, value in inputsY.items():
        axins.plot(inputsX, value, label = key)

    if y_label_list is None:
        zone_and_linked(ax, axins, x_start, x_end, inputsX , list(inputsY.values()), 'top')
    else:
        inputsYtmp = {}
        for key, value in inputsY.items():
            if key in y_label_list:
                inputsYtmp[key] = value
        zone_and_linked(ax, axins, x_start, x_end, inputsX , list(inputsYtmp.values()), 'top')
    plt.savefig(dir)

def line_multi_graphs_02(inputsX, inputsYs, label, dir, y_label_list = None, desc_info = None, tiny_win = True, sub_just = [0.275, 0.04, 0.995, 0.93]):#
    plt.rcParams['text.usetex']=True
    plt.rcParams['text.latex.preamble']=r'\makeatletter \newcommand*{\rom}[1]{\expandafter\@slowromancap\romannumeral #1@} \makeatother'
    fontsize = 60
    nrows = 1
    ncols = 2
    x_start = int(len(inputsX) * 0.9)
    x_end = len(inputsX) - 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(30, 15))
    for id, inputsY in enumerate(inputsYs):
        for key, value in inputsY.items():
            axs[id].plot(inputsX, value, label = key)
        axs[id].set_xlabel(label[0], fontsize=fontsize)
        axs[id].set_ylabel(label[1], fontsize=fontsize)
        axs[id].tick_params(axis='both', labelsize=fontsize)
        axs[id].set_title(r'(\rom{}) '.format(id+1) + desc_info[id]['air_craft'], fontsize = fontsize)

        if tiny_win:
            axins = axs[id].inset_axes((0.4, 0.45, 0.4, 0.5))
            for key, value in inputsY.items():
                axins.plot(inputsX, value, label = key)
            if y_label_list is None:
                zone_and_linked(axs[id], axins, x_start, x_end, inputsX , list(inputsY.values()), 'top')
            else:
                inputsYtmp = {}
                for key, value in inputsY.items():
                    if key in y_label_list:
                        inputsYtmp[key] = value
                zone_and_linked(axs[id], axins, x_start, x_end, inputsX , list(inputsYtmp.values()), 'top')

    fig.subplots_adjust(bottom=sub_just[0], left=sub_just[1], right=sub_just[2], top=sub_just[3], wspace=0.2, hspace=0.3)
    lines, labels = axs[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'lower center', ncol=3, borderaxespad=0, handlelength=4, frameon = False, fontsize=fontsize)
    plt.savefig(dir)

def line_graphs_03(inputsX, inputsY, label, dir, y_label_list = None):
    fig, ax = plt.subplots(1,1)
    for key, value in inputsY.items():
        ax.plot(inputsX, value, label = key)
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    # fig.spines['top'].set_visible(False)
    # fig.spines['right'].set_visible(False)
    plt.grid(c='w')
    plt.legend()
    plt.savefig(dir)