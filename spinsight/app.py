import holoviews as hv
import panel as pn
from pathlib import Path
import toml
from spinsight import MRIsimulator
from functools import partial
from datetime import datetime


def hide_show_button_callback(pane, event):
    if 'Show' in event.obj.name:
        pane.visible = True
        event.obj.name = event.obj.name.replace('Show', 'Hide')
    elif 'Hide' in event.obj.name:
        pane.visible = False
        event.obj.name = event.obj.name.replace('Hide', 'Show')


def load_button_callback(simulator, settings_file, event):
    print('Loading settings from file', settings_file)
    with open(settings_file, 'r') as f:
        settings = toml.load(f)
    simulator.set_params(settings)


def save_button_callback(simulator, settings_file, event):
    print('Saving settings to file', settings_file)
    settings = simulator.get_params()
    with open(settings_file, 'w') as f:
        toml.dump(settings, f)


def info_number(name, value, format, text_color):
    return pn.indicators.Number(default_color=text_color, name=name, format=format, font_size='12pt', title_size='12pt', value=value)


def info_string(name, value, text_color):
    return pn.indicators.String(default_color=text_color, name=name, font_size='12pt', title_size='12pt', value=value)


def get_app(dark_mode=True, settings_filestem='', start_time=datetime.now(), lazy_sliders=True):
    pn.config.theme = 'dark' if dark_mode else 'default'
    pn.config.throttled = lazy_sliders
    text_color = 'white' if dark_mode else 'black' # needed for pn.indicators.Number which doesn't respect pn.config.theme

    settings_file = Path(settings_filestem).with_suffix('.toml') if bool(settings_filestem) else Path('')

    simulator = MRIsimulator.MRIsimulator(name='')
    title = 'SpinSight MRI simulator'
    author = '*Written by [Johan Berglund](mailto:johan.berglund@akademiska.se), Ph.D.*'
    try:
        version = f', v {toml.load(Path(__file__).parent.parent / "pyproject.toml")["project"]["version"]}'
    except (FileNotFoundError, KeyError):
        version = ''
    
    discrete_slider_params = ['TR', 'TE', 'FA', 'TI', 'FOV_F', 'FOV_P', 'phase_oversampling', 'matrix_F', 'matrix_P', 'recon_matrix_F', 'recon_matrix_P', 'voxel_F', 'voxel_P', 'recon_voxel_F', 'recon_voxel_P', 'slice_thickness', 'pixel_bandwidth_param', 'FOV_bandwidth', 'FW_shift', 'EPI_factor']
    param_panels = {name: pn.panel(simulator.param, parameters=params, widgets={p: pn.widgets.DiscreteSlider for p in params if p in discrete_slider_params}, name=name) for name, params in [
        ('Settings', ['object', 'field_strength', 'parameter_style']),
        ('Contrast', ['FatSat', 'TR', 'TE', 'FA', 'TI']),
        ('Geometry', ['trajectory', 'frequency_direction', 'FOV_F', 'FOV_P', 'phase_oversampling', 'radial_factor', 'voxel_F', 'voxel_P', 'matrix_F', 'matrix_P', 'recon_voxel_F', 'recon_voxel_P', 'recon_matrix_F', 'recon_matrix_P', 'slice_thickness']),
        ('Sequence', ['sequence_type', 'pixel_bandwidth_param', 'FOV_bandwidth', 'FW_shift', 'NSA', 'partial_Fourier', 'turbo_factor', 'EPI_factor']),
        ('Post-processing', ['homodyne', 'do_apodize', 'apodization_alpha', 'do_zerofill']),
    ]}

    info_pane = pn.Row(info_number(name='Relative SNR', format='{value:.0f}%', value=simulator.param.relative_SNR, text_color=text_color),
                      info_string(name='Scan time', value=simulator.param.scantime, text_color=text_color),
                      info_number(name='Fat/water shift', format='{value:.2f} pixels', value=simulator.param.FW_shift, text_color=text_color),
                      info_number(name='Bandwidth', format='{value:.0f} Hz/pixel', value=simulator.param.pixel_bandwidth_param, text_color=text_color))
    shot_angle_info = info_number(name='Angle', format='{value:.0f}°', value=simulator.param.spoke_angle, text_color=text_color) 
    num_shots_info = info_number(name='# shots', format='{value:.0f}', value=simulator.param.num_shots, text_color=text_color)
    def update_num_shots_label(event): 
        num_shots_info.name = f'# {event.new}s'
    simulator.param.watch(update_num_shots_label, 'shot_label')
    
    dmap_kspace = pn.Column(hv.DynamicMap(simulator.display_kspace) * simulator.k_line, 
                           # simulator.param.kspace_type, 
                           pn.Row(simulator.param.show_processed_kspace, simulator.param.kspace_exponent), 
                           visible=False)
    dmap_MR_image = hv.DynamicMap(simulator.display_image)
    dmap_sequence = pn.Column(hv.DynamicMap(simulator.display_sequence_plot), pn.Row(simulator.param.shot, shot_angle_info, num_shots_info), visible=False)
    load_button = pn.widgets.Button(name='Load settings', visible=settings_file.is_file())
    load_button.on_click(partial(load_button_callback, simulator, settings_file))
    save_button = pn.widgets.Button(name='Save settings', visible=settings_file.is_file())
    save_button.on_click(partial(save_button_callback, simulator, settings_file))
    sequence_button = pn.widgets.Button(name='Show sequence')
    sequence_button.on_click(partial(hide_show_button_callback, dmap_sequence))
    kspace_button = pn.widgets.Button(name='Show k-space')
    kspace_button.on_click(partial(hide_show_button_callback, dmap_kspace))
    reset_SNR_button = pn.widgets.Button(name='Set reference SNR')
    reset_SNR_button.on_click(simulator.set_reference_SNR)
    dashboard = pn.Column(
        pn.Row(
            pn.Column(
                pn.Row(
                    pn.Column(
                        pn.Row(sequence_button, kspace_button), 
                        param_panels['Sequence'],
                        param_panels['Contrast']
                    ), 
                    pn.Column(
                        param_panels['Geometry']
                    )
                )
            ), 
            pn.Column(
                dmap_MR_image, 
                pn.Column(
                    # simulator.param.image_type, 
                    pn.Row(reset_SNR_button, simulator.param.show_FOV), 
                    simulator.param.reference_tissue, 
                    info_pane
                )
            ), 
            pn.Column(
                dmap_kspace,
                param_panels['Post-processing']
            )
        ), 
        dmap_sequence, 
        pn.Column(pn.pane.Markdown(author),
                  pn.pane.Markdown(f'*(server started {start_time: %Y-%m-%d %H:%M:%S}{version})*', styles={'color': 'gray'}),
                  height=10),
        sizing_mode='stretch_both'
    )

    template = pn.template.FastListTemplate(
        title=title,
        theme='dark' if dark_mode else 'default',
        theme_toggle=False,
        header_background='#262626',
        modal=[pn.Column()],
        collapsed_sidebar=True
    )
    template.main.append(dashboard)
    template.sidebar.append(
        pn.Column(
            pn.Row(load_button, save_button), 
            param_panels['Settings'], 
        )
    )
    return template
