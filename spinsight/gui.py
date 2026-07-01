import param
import panel as pn
import holoviews as hv
from functools import partial
import toml
from spinsight import styles
from spinsight.hover_manager import HoverManager
from spinsight.params import PARAMS
from spinsight.styles import INFO_FONT_SIZE, INFO_TITLE_SIZE, INFO_TEXT_COLOR


def hide_show_button_callback(pane, event):
    if 'Show' in event.obj.name:
        pane.visible = True
        event.obj.name = event.obj.name.replace('Show', 'Hide')
    elif 'Hide' in event.obj.name:
        pane.visible = False
        event.obj.name = event.obj.name.replace('Hide', 'Show')


def load_button_callback(controller, settings_file, event):
    print('Loading settings from file', settings_file)
    with open(settings_file, 'r') as f:
        settings = toml.load(f)
    controller.set_input_params(settings)


def save_button_callback(controller, settings_file, event):
    print('Saving settings to file', settings_file)
    settings = controller.get_input_params()
    with open(settings_file, 'w') as f:
        toml.dump(settings, f)


def info(indicator_func, **kwargs):
    return indicator_func(font_size=INFO_FONT_SIZE, title_size=INFO_TITLE_SIZE, **kwargs)


class GUI(param.Parameterized):
    
    image = param.Parameter()
    kspace = param.Parameter()
    sequence_plot = param.Parameter()
    
    def __init__(self, controller, settings_file, start_time, version, lazy_sliders, dark_mode, **params):
        
        super().__init__(**params)

        self.controller = controller
        self.version = version
        self.settings_file = settings_file
        self.start_time = start_time
        
        self.hover = HoverManager()

        pn.config.throttled = lazy_sliders

        self.dashboard = self.make_layout('SpinSight MRI simulator', dark_mode)
    
    @param.depends('image')
    def display_image(self):
        return self.image
    
    @param.depends('kspace')
    def display_kspace(self):
        return self.kspace
    
    @param.depends('sequence_plot')
    def display_sequence_plot(self):
        return self.sequence_plot
        
    def make_layout(self, title, dark_mode):
        dashboard = styles.panel_template(title, dark_mode)

        author = '*Written by [Johan Berglund](mailto:johan.berglund@akademiska.se), Ph.D.*'
        
        shot_angle_info = info(pn.indicators.Number, name='Angle', format='{value:.0f}°', value=self.controller.param.spoke_angle, default_color=INFO_TEXT_COLOR) 
        num_shots_info = info(pn.indicators.Number, name='# shots', format='{value:.0f}', value=self.controller.param.num_shots, default_color=INFO_TEXT_COLOR)
        def update_num_shots_label(event): 
            num_shots_info.name = f'# {event.new}s'
        self.controller.param.watch(update_num_shots_label, 'shot_label')
        
        dmap_kspace = pn.Column(hv.DynamicMap(self.display_kspace) * self.hover.k_line, 
                            # self.controller.input.param.kspace_type, 
                            pn.Row(self.controller.input.param.show_processed_kspace, self.controller.input.param.kspace_exponent), 
                            visible=False)
        dmap_MR_image = hv.DynamicMap(self.display_image)
        dmap_sequence = pn.Column(hv.DynamicMap(self.display_sequence_plot), pn.Row(self.controller.input.param.shot_ui, shot_angle_info, num_shots_info, self.controller.input.param.signal_exponent), visible=False)
        load_button = pn.widgets.Button(name='Load settings', visible=self.settings_file.is_file())
        load_button.on_click(partial(load_button_callback, self.controller, self.settings_file))
        save_button = pn.widgets.Button(name='Save settings', visible=self.settings_file.is_file())
        save_button.on_click(partial(save_button_callback, self.controller, self.settings_file))
        sequence_button = pn.widgets.Button(name='Show sequence')
        sequence_button.on_click(partial(hide_show_button_callback, dmap_sequence))
        kspace_button = pn.widgets.Button(name='Show k-space')
        kspace_button.on_click(partial(hide_show_button_callback, dmap_kspace))
        reset_SNR_button = pn.widgets.Button(name='Set reference SNR')
        reset_SNR_button.on_click(self.controller.set_reference_SNR)
        maindash = pn.Column(
            pn.Row(
                pn.Column(
                    pn.Row(
                        pn.Column(
                            pn.Row(sequence_button, kspace_button), 
                            self.make_param_panel('Sequence'),
                            self.make_param_panel('Contrast')
                        ), 
                        pn.Column(
                            self.make_param_panel('Geometry')
                        )
                    )
                ), 
                pn.Column(
                    dmap_MR_image, 
                    pn.Column(
                        # self.controller.input.param.image_type, 
                        self.controller.input.param.show_FOV,
                        self.make_info_panel(reset_SNR_button)
                    )
                ), 
                pn.Column(
                    dmap_kspace,
                    self.make_param_panel('Post-processing')
                )
            ), 
            dmap_sequence, 
            pn.Column(pn.pane.Markdown(author),
                    pn.pane.Markdown(f'*(server started {self.start_time: %Y-%m-%d %H:%M:%S}{self.version})*', styles={'color': 'gray'}),
                    height=10),
            sizing_mode='stretch_both'
        )

        dashboard.main.append(maindash)
        dashboard.sidebar.append(
            pn.Column(
                pn.Row(load_button, save_button), 
                self.make_param_panel('Settings')
            )
        )
        return dashboard
    
    def make_info_panel(self, reset_SNR_button):
        return pn.Column(
            pn.Row(self.controller.input.param.reference_tissue, reset_SNR_button),
            pn.Row(
                info(pn.indicators.Number, name='Relative SNR', format='{value:.0f}%', value=self.controller.param.relative_SNR, default_color=INFO_TEXT_COLOR),
                info(pn.indicators.String, name='Scan time', value=self.controller.param.scantime, default_color=INFO_TEXT_COLOR),
                info(pn.indicators.Number, name='Fat/water shift', format='{value:.2f} pixels', value=self.controller.input.param.FW_shift, default_color=INFO_TEXT_COLOR),
                info(pn.indicators.Number, name='Bandwidth', format='{value:.0f} Hz/pixel', value=self.controller.input.param.pixel_bandwidth_ui, default_color=INFO_TEXT_COLOR)
            )
        )

    def make_param_panel(self, name):
        params = [par for par in PARAMS if PARAMS[par].group == name]
        return pn.panel(self.controller.input.param, parameters=params, widgets={p: PARAMS[p].widget for p in params}, name=name)
    
    def view(self):
        return self.dashboard