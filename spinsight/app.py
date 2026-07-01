import holoviews as hv
import panel as pn
from pathlib import Path
import toml
from spinsight import styles, simulator
from spinsight.params import PARAMS
from spinsight.styles import INFO_FONT_SIZE, INFO_TITLE_SIZE
from spinsight.controller import Controller
from spinsight.dashboard import Dashboard
from functools import partial
from datetime import datetime


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


def get_version():
    try:
        return f', v {toml.load(Path(__file__).parent.parent / "pyproject.toml")["project"]["version"]}'
    except (FileNotFoundError, KeyError):
        return ''


def make_param_panel(name, controller):
    params = [par for par in PARAMS if PARAMS[par].group == name]
    return pn.panel(controller.input.param, parameters=params, widgets={p: PARAMS[p].widget for p in params}, name=name)


def get_app(dark_mode=True, settings_filestem='', start_time=datetime.now(), lazy_sliders=True):
    pn.config.throttled = lazy_sliders
    text_color = styles.text_color(dark_mode)

    settings_file = Path(settings_filestem).with_suffix('.toml') if bool(settings_filestem) else Path('')

    controller = Controller()
    dashboard = Dashboard()
    graph = simulator.make_graph(controller, dashboard)
    controller.add_input_watchers(graph)
    controller.set_reference_SNR()

    title = 'SpinSight MRI simulator'
    author = '*Written by [Johan Berglund](mailto:johan.berglund@akademiska.se), Ph.D.*'
    version = get_version()
    
    info_pane = pn.Row(info(pn.indicators.Number, name='Relative SNR', format='{value:.0f}%', value=controller.param.relative_SNR, default_color=text_color),
                      info(pn.indicators.String, name='Scan time', value=controller.param.scantime, default_color=text_color),
                      info(pn.indicators.Number, name='Fat/water shift', format='{value:.2f} pixels', value=controller.input.param.FW_shift, default_color=text_color),
                      info(pn.indicators.Number, name='Bandwidth', format='{value:.0f} Hz/pixel', value=controller.input.param.pixel_bandwidth_ui, default_color=text_color))
    shot_angle_info = info(pn.indicators.Number, name='Angle', format='{value:.0f}°', value=controller.param.spoke_angle, default_color=text_color) 
    num_shots_info = info(pn.indicators.Number, name='# shots', format='{value:.0f}', value=controller.param.num_shots, default_color=text_color)
    def update_num_shots_label(event): 
        num_shots_info.name = f'# {event.new}s'
    controller.param.watch(update_num_shots_label, 'shot_label')
    
    dmap_kspace = pn.Column(hv.DynamicMap(dashboard.display_kspace) * dashboard.hover.k_line, 
                           # controller.input.param.kspace_type, 
                           pn.Row(controller.input.param.show_processed_kspace, controller.input.param.kspace_exponent), 
                           visible=False)
    dmap_MR_image = hv.DynamicMap(dashboard.display_image)
    dmap_sequence = pn.Column(hv.DynamicMap(dashboard.display_sequence_plot), pn.Row(controller.input.param.shot_ui, shot_angle_info, num_shots_info, controller.input.param.signal_exponent), visible=False)
    load_button = pn.widgets.Button(name='Load settings', visible=settings_file.is_file())
    load_button.on_click(partial(load_button_callback, controller, settings_file))
    save_button = pn.widgets.Button(name='Save settings', visible=settings_file.is_file())
    save_button.on_click(partial(save_button_callback, controller, settings_file))
    sequence_button = pn.widgets.Button(name='Show sequence')
    sequence_button.on_click(partial(hide_show_button_callback, dmap_sequence))
    kspace_button = pn.widgets.Button(name='Show k-space')
    kspace_button.on_click(partial(hide_show_button_callback, dmap_kspace))
    reset_SNR_button = pn.widgets.Button(name='Set reference SNR')
    reset_SNR_button.on_click(controller.set_reference_SNR)
    maindash = pn.Column(
        pn.Row(
            pn.Column(
                pn.Row(
                    pn.Column(
                        pn.Row(sequence_button, kspace_button), 
                        make_param_panel('Sequence', controller),
                        make_param_panel('Contrast', controller)
                    ), 
                    pn.Column(
                        make_param_panel('Geometry', controller)
                    )
                )
            ), 
            pn.Column(
                dmap_MR_image, 
                pn.Column(
                    # controller.input.param.image_type, 
                    pn.Row(reset_SNR_button, controller.input.param.show_FOV), 
                    controller.input.param.reference_tissue, 
                    info_pane
                )
            ), 
            pn.Column(
                dmap_kspace,
                make_param_panel('Post-processing', controller)
            )
        ), 
        dmap_sequence, 
        pn.Column(pn.pane.Markdown(author),
                  pn.pane.Markdown(f'*(server started {start_time: %Y-%m-%d %H:%M:%S}{version})*', styles={'color': 'gray'}),
                  height=10),
        sizing_mode='stretch_both'
    )

    template = styles.panel_template(title, dark_mode)
    template.main.append(maindash)
    template.sidebar.append(
        pn.Column(
            pn.Row(load_button, save_button), 
            make_param_panel('Settings', controller)
        )
    )
    return template