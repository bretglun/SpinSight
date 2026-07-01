import holoviews as hv
from holoviews import streams
import numpy as np
import math
from pathlib import Path
from bokeh.models import HoverTool, CustomJS, ColumnDataSource


def get_k_on_interval(interval, k_trajectory):
    t = np.arange(*interval[[0, -1]], k_trajectory['dt'])
    kx = np.interp(t, k_trajectory['t'], k_trajectory['kx'])
    ky = np.interp(t, k_trajectory['t'], k_trajectory['ky'])
    return zip(kx, ky)


def arrow(coords):
    angle = -np.degrees(math.atan2(coords[-1][0]-coords[-2][0], coords[-1][1]-coords[-2][1])) if len(coords)>1 else 0
    return hv.Curve(coords) * hv.Points(coords[-1]).opts(angle=angle, marker='triangle')


class HoverManager():
    
    def __init__(self):

        self.k_trajectory = None
        self.objects = {'frequency': [], 'phase': [], 'RF': [], 'signal': []}

        arrow_stream = streams.Stream.define('arrow', coords=[None])
        self.k_line = hv.DynamicMap(arrow, streams=[arrow_stream()])

        self.hover_index = ColumnDataSource({'index': [], 'board': []})
        self.hover_index.on_change('data', self.update_k_line_coords)
    
    def get_hover_tool(self, board, attributes):
        with open(Path(__file__).parent / 'hoverCallback.js', 'r') as file:
            hover_callback = CustomJS(args={'hover_index': self.hover_index, 'board': board}, code=file.read())
        if board == 'slice':
            hover_callback = None
        return HoverTool(tooltips=[(attr, f'@{attr}') for attr in attributes], attachment='below', callback=hover_callback)

    def update_k_line_coords(self, attr, old, hover_index):
        if len(hover_index['index']) == 0:
            self.k_line.event(coords=[None])
            return
        board = hover_index['board'][0]
        index = hover_index['index'][0]
        object = self.objects[board][index]
        self.k_line.event(coords=list(get_k_on_interval(object['time'][[0, -1]], self.k_trajectory)))