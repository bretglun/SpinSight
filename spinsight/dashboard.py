import param
from spinsight.hover_manager import HoverManager


class Dashboard(param.Parameterized):
    
    image = param.Parameter()
    kspace = param.Parameter()
    sequence_plot = param.Parameter()
    
    def __init__(self, **params):
        
        super().__init__(**params)
        self.hover = HoverManager()
    
    @param.depends('image')
    def display_image(self):
        return self.image
    
    @param.depends('kspace')
    def display_kspace(self):
        return self.kspace
    
    @param.depends('sequence_plot')
    def display_sequence_plot(self):
        return self.sequence_plot