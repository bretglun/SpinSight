import param
import panel as pn
import holoviews as hv
from functools import partial
import toml
from spinsight import styles
from spinsight.controller import Controller
from spinsight.params import PARAMS
from spinsight.styles import INFO_FONT_SIZE, INFO_TITLE_SIZE, INFO_TEXT_COLOR, IMAGE_HEIGHT, IMAGE_WIDTH
import asyncio


def hide_show_button_callback(pane, event):
    if 'Show' in event.obj.name:
        pane.visible = True
        event.obj.name = event.obj.name.replace('Show', 'Hide')
    elif 'Hide' in event.obj.name:
        pane.visible = False
        event.obj.name = event.obj.name.replace('Hide', 'Show')


class GUI(param.Parameterized):
    
    image = param.Parameter()
    kspace = param.Parameter()
    sequence_plot = param.Parameter()

    spoke_angle = param.String(label='Angle')
    num_shots = param.String(label='# shots')
    relative_SNR = param.String(label='Relative SNR')
    scantime = param.String(label='Scan time')
    FW_shift = param.String(label='Fat/water shift')
    pixel_bandwidth = param.String(label='Bandwidth')
    
    def __init__(self, settings_file, start_time, version, lazy_sliders, dark_mode, **params):
        
        super().__init__(**params)
        
        self.version = version
        self.settings_file = settings_file
        self.start_time = start_time

        self.make_indicators()
        
        self.progress_bar = self.make_progress_bar()
        self.controller = Controller(self)

        pn.config.throttled = lazy_sliders

        self.dashboard = self.make_layout('SpinSight MRI simulator', dark_mode)

        pn.state.onload(self.initialize)

    async def initialize(self):
        self.show_progress_bar(self.controller.input.object)
        await asyncio.sleep(0)  # let Panel flush the modal-open update
        await asyncio.to_thread(self.controller.sync_with_graph)
        self.hide_progress_bar()
        self.initialize_plots()
    
    @param.depends('image')
    def display_image(self):
        return self.image
    
    @param.depends('kspace')
    def display_kspace(self):
        return self.kspace
    
    @param.depends('sequence_plot')
    def display_sequence_plot(self):
        return self.sequence_plot
        
    def make_progress_bar(self):
        return tuple(pn.widgets.Tqdm(width=1000, height=70, margin=(0, 0, 0, indent), text_pane=pn.pane.Str(styles={'color': INFO_TEXT_COLOR})) for indent in (10, 90))

    def make_progress_bar_modal(self):
        return pn.Column('', *self.progress_bar)
    
    def set_progress_bar_title(self, phantom_name=''):
        self.dashboard.modal[0][0] = f'# Preparing *k*-space for "{phantom_name}" phantom...\n*(first-time loading may take several minutes)*'

    def make_layout(self, title, dark_mode):
        dashboard = styles.panel_template(title, dark_mode)
        
        image_panel = self.make_image_panel()
        kspace_panel = self.make_kspace_panel()
        sequence_plot_panel = self.make_sequence_plot_panel()
        sequence_button = self.make_sequence_button(sequence_plot_panel)
        kspace_button = self.make_kspace_button(kspace_panel)
        load_button = self.make_load_button()
        save_button = self.make_save_button()
        reset_SNR_button = self.make_reset_SNR_button()
        info_panel = self.make_info_panel(reset_SNR_button)
        settings_params_panel = self.make_param_group_panel('Settings')
        sequence_params_panel = self.make_param_group_panel('Sequence')
        contrast_params_panel = self.make_param_group_panel('Contrast')
        geometry_params_panel = self.make_param_group_panel('Geometry')
        post_processing_params_panel = self.make_param_group_panel('Post-processing')
        footer_panel = self.make_footer_panel()

        dashboard.sidebar.append(self.make_sidebar(
            load_button, 
            save_button, 
            settings_params_panel
            )
        )
        dashboard.main.append(self.make_main_page(
            sequence_button, 
            kspace_button, 
            sequence_params_panel, 
            contrast_params_panel, 
            geometry_params_panel, 
            image_panel, 
            info_panel, 
            kspace_panel, 
            post_processing_params_panel, 
            sequence_plot_panel, 
            footer_panel
            )
        )

        dashboard.modal[0] = self.make_progress_bar_modal()

        return dashboard

    def make_sidebar(self, load_button, save_button, settings_params_panel):
        return pn.Column(
            pn.Row(load_button, save_button), 
            settings_params_panel
        )
    
    def make_main_page(self, sequence_button, kspace_button, sequence_params_panel, contrast_params_panel, geometry_params_panel, image_panel, info_panel, kspace_panel, post_processing_params_panel, sequence_plot_panel, footer_panel):
        return pn.Column(
            pn.Row(
                pn.Column(
                    pn.Row(
                        pn.Column(
                            pn.Row(sequence_button, kspace_button), 
                            sequence_params_panel,
                            contrast_params_panel
                        ),
                        geometry_params_panel
                    )
                ), 
                pn.Column(
                    image_panel,
                    info_panel
                ), 
                pn.Column(
                    kspace_panel,
                    post_processing_params_panel
                )
            ), 
            sequence_plot_panel, 
            footer_panel,
            sizing_mode='stretch_both'
        )

    def make_image_panel(self):
        return pn.Column(
            self.make_image_container(), 
            pn.Row(
                self.controller.input.param.image_mode, 
                pn.Param(self.controller.input.param, parameters=['show_FOV'], show_name=False, margin=(30, 0, 0, 0))
            )
        )
    
    def make_kspace_panel(self):
        return pn.Column(
            self.make_kspace_container(), 
            pn.Row(
                self.controller.input.param.kspace_mode, 
                pn.Param(self.controller.input.param, parameters=['kspace_exponent'], show_name=False, width=150)
                ),
            self.controller.input.param.show_processed_kspace,
            visible=False
        )
    
    def make_sequence_plot_panel(self):
        def update_num_shots_label(event): 
            self.indicators['num_shots'].name = f'# {event.new}s'
        self.controller.param.watch(update_num_shots_label, 'shot_label')
        return pn.Column(
            self.make_sequence_plot_container(), 
            pn.Row(
                self.make_param_panel('shot'),
                self.indicators['spoke_angle'], 
                self.indicators['num_shots'], 
                self.make_param_panel('signal_exponent')
            ),
            visible=False
        )

    def make_image_container(self):
        spinner = pn.indicators.LoadingSpinner(label='Preparing image', value=True, size=50, align='center')
        self.image_container = pn.Column(pn.Spacer(height=IMAGE_HEIGHT//2), spinner, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
        return self.image_container

    def make_kspace_container(self):
        spinner = pn.indicators.LoadingSpinner(label='Preparing k-space', value=True, size=50, align='center')
        self.kspace_container = pn.Column(pn.Spacer(height=IMAGE_HEIGHT//2), spinner, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
        return self.kspace_container

    def make_sequence_plot_container(self):
        spinner = pn.indicators.LoadingSpinner(label='Preparing sequence plot', value=True, size=50, align='center')
        self.sequence_plot_container = pn.Column(spinner)
        return self.sequence_plot_container

    def initialize_plots(self):
        self.image_container[:] = [hv.DynamicMap(self.display_image)]
        self.kspace_container[:] = [hv.DynamicMap(self.display_kspace) * self.controller.from_graph('hover_manager').k_line]
        self.sequence_plot_container[:] = [hv.DynamicMap(self.display_sequence_plot)]

    def make_info_panel(self, reset_SNR_button):
        return pn.Column(
            pn.Row(
                self.controller.input.param.reference_tissue, 
                pn.Column(pn.Spacer(height=20), reset_SNR_button),
            ),
            pn.Row(*(self.indicators[par] for par in ['relative_SNR', 'scantime', 'FW_shift', 'pixel_bandwidth']))
        )

    def show_progress_bar(self, phantom_name=''):
        if not hasattr(self, 'dashboard'):
            return
        self.set_progress_bar_title(phantom_name)
        self.dashboard.open_modal()
        
    def hide_progress_bar(self):
        if not hasattr(self, 'dashboard'):
            return
        self.dashboard.close_modal()
    
    def indicator(self, par_name):
        par = self.param[par_name]
        return pn.indicators.String(value=par, name=par.label, font_size=INFO_FONT_SIZE, title_size=INFO_TITLE_SIZE, default_color=INFO_TEXT_COLOR)

    def make_indicators(self):
        self.indicators = {par_name: self.indicator(par_name) for par_name in ['spoke_angle', 'num_shots', 'relative_SNR', 'scantime', 'FW_shift', 'pixel_bandwidth']}
    
    def set_indicator_visibility(self, par_name, visible):
        self.indicators[par_name].visible = visible

    def make_footer_panel(self):
        author = '*Written by [Johan Berglund](mailto:johan.berglund@akademiska.se), Ph.D.*'
        start_time = f'{self.start_time: %Y-%m-%d %H:%M:%S}'
        return pn.Column(
            pn.pane.Markdown(author, styles={'color': INFO_TEXT_COLOR}),
            pn.pane.Markdown(f'*(server started {start_time}{self.version})*', styles={'color': INFO_TEXT_COLOR}), height=10)

    def make_param_panel(self, param):
        return pn.panel(self.controller.input.param[param], widgets={param: PARAMS[param].widget})
    
    def make_param_group_panel(self, group):
        params = [par for par in PARAMS if PARAMS[par].group == group]
        return pn.panel(self.controller.input.param, parameters=params, widgets={par: PARAMS[par].widget for par in params}, name=group)
    
    def make_sequence_button(self, sequence_plot_panel):
        sequence_button = pn.widgets.Button(name='Show sequence')
        sequence_button.on_click(partial(hide_show_button_callback, sequence_plot_panel))
        return sequence_button
    
    def make_kspace_button(self, kspace_panel):
        kspace_button = pn.widgets.Button(name='Show k-space')
        kspace_button.on_click(partial(hide_show_button_callback, kspace_panel))
        return kspace_button
    
    def make_load_button(self):
        load_button = pn.widgets.Button(name='Load settings', visible=self.settings_file.is_file())
        load_button.on_click(self.load_button_callback)
        return load_button
    
    def make_save_button(self):
        save_button = pn.widgets.Button(name='Save settings', visible=self.settings_file.is_file())
        save_button.on_click(self.save_button_callback)
        return save_button
    
    def load_button_callback(self, event):
        print('Loading settings from file', self.settings_file)
        with open(self.settings_file, 'r') as f:
            settings = toml.load(f)
        self.controller.set_input_params(settings)
    
    def save_button_callback(self, event):
        print('Saving settings to file', self.settings_file)
        settings = self.controller.get_input_params()
        with open(self.settings_file, 'w') as f:
            toml.dump(dict(sorted(settings.items())), f)
    
    def make_reset_SNR_button(self):
        reset_SNR_button = pn.widgets.Button(name='Set reference SNR')
        reset_SNR_button.on_click(self.controller.set_reference_SNR)
        return reset_SNR_button
    
    def view(self):
        return self.dashboard