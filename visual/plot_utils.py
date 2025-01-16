import warnings
from matplotlib import pyplot as plt
from visual.plottools  import *
from matplotlib.widgets import Slider
import matplotlib.ticker as mtick

def plot_loss(x, y):
    plt.plot(x, y)
    plt.show()

#  https://blog.csdn.net/mighty13/article/details/113062528

class mainWin:
    def __init__(self, consist_update = True, line_config = [0, 100000], auxiliary_line = None, auxiliary_points = None) -> None:
        self.colorlist = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.line_config = [i * 0.3048 for i in line_config]

        self.consist_update = consist_update
        self.fig = plt.figure()
        self.auxiliary_line = auxiliary_line
        self.auxiliary_points = auxiliary_points
        
        if self.consist_update:
            plt.ion()
        
    def draw(self, curobs):
        if self.consist_update:
            plt.clf()
        
        self.curobs = curobs

        # self.path_positions = [[0, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 2, 2]]
        self.path_positions = self.curobs[:, :3]
        self.path_rotations = self.curobs[:, 3:]
               
        self.x_border = [np.min(self.path_positions[:,0]), np.max(self.path_positions[:,0])]
        self.y_border = [np.min(self.path_positions[:,1]), np.max(self.path_positions[:,1])]
        self.z_border = [np.min(self.path_positions[:,2]), np.max(self.path_positions[:,2])]
        if self.auxiliary_line is not None:
            aux_x_border = [np.min(self.auxiliary_line[:,0]), np.max(self.auxiliary_line[:,0])]
            aux_y_border = [np.min(self.auxiliary_line[:,1]), np.max(self.auxiliary_line[:,1])]
            aux_z_border = [np.min(self.auxiliary_line[:,2]), np.max(self.auxiliary_line[:,2])]
            
            self.x_border = [min(self.x_border[0], aux_x_border[0]), max(self.x_border[1], aux_x_border[1])]
            self.y_border = [min(self.y_border[0], aux_y_border[0]), max(self.y_border[1], aux_y_border[1])]
            self.z_border = [min(self.z_border[0], aux_z_border[0]), max(self.z_border[1], aux_z_border[1])]
        
        self.positions = [(self.path_positions[0, 0], self.path_positions[0, 1], self.path_positions[0, 2])]
        self.rotation = [self.path_rotations[0, 0], self.path_rotations[0, 1], self.path_rotations[0, 2]]
        self.sizes = [((self.x_border[1]-self.x_border[0]) * 0.1, (self.y_border[1]-self.y_border[0]) * 0.1, (self.z_border[1]-self.z_border[0]) * 0.1)]
        self.colors = ["Gray"]

        # self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_aspect('equal')

        plotPoint(self.positions, self.ax)
        if self.auxiliary_points is not None:
            plotPoint([item['pos'] for item in self.auxiliary_points], self.ax, [item['color'] for item in self.auxiliary_points])
        plotCurve(self.path_positions, self.ax, self.line_config, self.colorlist)
        # if self.auxiliary_line is not None:
        #     plotCurve01(self.auxiliary_line, self.ax, color='r')
        pc = plotPlane(self.positions, self.rotation, self.sizes, colors=self.colors, edgecolor="k")
        self.ax.add_collection3d(pc)


        axcolor = 'lightgoldenrodyellow'
        axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        self.sfreq = Slider(axfreq, 'Freq', 0.0, 100.0,valfmt='% .2f', valinit=0, valstep=0.01)


        self.sfreq.on_changed(self.update)
        self.sfreq.reset()
        self.sfreq.set_val(0.0)

        warnings.filterwarnings('ignore')
        fmt='%d'
        ticks = mtick.FormatStrFormatter(fmt)
        self.ax.xaxis.set_major_formatter(ticks)
        self.ax.yaxis.set_major_formatter(ticks)
        self.ax.zaxis.set_major_formatter(ticks)
        
        # self.ax.set_xlim(self.x_border)
        # self.ax.set_ylim(self.y_border)
        # self.ax.set_zlim(self.z_border)
        self.ax.set_xlabel('Latitude/m')
        self.ax.set_ylabel('Longitude/m')
        self.ax.set_zlabel('Altitude/m')
        self.ax.plot([self.x_border[0], self.x_border[1]], [self.y_border[0], self.y_border[0]], [self.z_border[0], self.z_border[0]],color=(1,0,0,0.125))
        self.ax.plot([self.x_border[0], self.x_border[0]], [self.y_border[0], self.y_border[1]], [self.z_border[0], self.z_border[0]],color=(0,1,0,0.125))
        self.ax.plot([self.x_border[0], self.x_border[0]], [self.y_border[0], self.y_border[0]], [self.z_border[0], self.z_border[1]],color=(0,0,1,0.125))
        
        if self.consist_update:
            plt.pause(0.1)
            plt.ioff()
        
        if not self.consist_update:
            plt.show()
        

    def update(self, val):
        freq = self.sfreq.val
        self.ax.clear()
        
        frame_id = int((len(self.curobs) - 1) * freq / 100)
        self.positions = [(self.path_positions[frame_id, 0], self.path_positions[frame_id, 1], self.path_positions[frame_id, 2])]
        self.rotation = [self.path_rotations[frame_id, 0], self.path_rotations[frame_id, 1], self.path_rotations[frame_id, 2]]
        
        pc = plotPlane(self.positions, self.rotation, self.sizes, colors=self.colors, edgecolor="k")  # edgecolor边缘线要不要
        self.ax.add_collection3d(pc)
        plotPoint(self.positions, self.ax)
        if self.auxiliary_points is not None:
            plotPoint([item['pos'] for item in self.auxiliary_points], self.ax, [item['color'] for item in self.auxiliary_points])
        plotCurve(self.path_positions, self.ax, self.line_config, self.colorlist)
        if self.auxiliary_line is not None:
            plotCurve01(self.auxiliary_line, self.ax, color='r')
        
        warnings.filterwarnings('ignore')
        fmt='%d'
        ticks = mtick.FormatStrFormatter(fmt)
        self.ax.xaxis.set_major_formatter(ticks)
        self.ax.yaxis.set_major_formatter(ticks)
        self.ax.zaxis.set_major_formatter(ticks)
        
        # self.ax.set_xlim(self.x_border)
        # self.ax.set_ylim(self.y_border)
        # self.ax.set_zlim(self.z_border)
        self.ax.set_xlabel('Latitude/m')
        self.ax.set_ylabel('Longitude/m')
        self.ax.set_zlabel('Altitude/m')
        self.ax.plot([self.x_border[0], self.x_border[1]], [self.y_border[0], self.y_border[0]], [self.z_border[0], self.z_border[0]],color=(1,0,0,0.125))
        self.ax.plot([self.x_border[0], self.x_border[0]], [self.y_border[0], self.y_border[1]], [self.z_border[0], self.z_border[0]],color=(0,1,0,0.125))
        self.ax.plot([self.x_border[0], self.x_border[0]], [self.y_border[0], self.y_border[0]], [self.z_border[0], self.z_border[1]],color=(0,0,1,0.125))
        self.fig.canvas.draw_idle()

    def plot_auxiliary_line(self):
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_aspect('equal')
        plotCurve01(self.auxiliary_line, self.ax, color='r')
        plt.show()
# mainWin()