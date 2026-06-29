import holoviews as hv
import panel as pn

hv.extension('bokeh')


BOARD_COLORS = {
    'frequency': 'cadetblue',
    'phase': 'cadetblue',
    'slice': 'cadetblue',
    'RF': 'red',
    'signal': 'orange',
    'ADC': 'peru',
}

hv.opts.defaults(hv.opts.Image(width=500, height=500, invert_yaxis=False, toolbar='below', cmap='gray', aspect='equal'))
hv.opts.defaults(hv.opts.HLine(line_width=1.5, line_color='gray'))
hv.opts.defaults(hv.opts.VSpan(color='orange', fill_alpha=.1, hover_fill_alpha=.8, default_tools=[]))
hv.opts.defaults(hv.opts.Rectangles(color=BOARD_COLORS['ADC'], line_color=BOARD_COLORS['ADC'], fill_alpha=.1, line_alpha=.3, hover_fill_alpha=.8, default_tools=[]))
hv.opts.defaults(hv.opts.Box(line_width=3))
hv.opts.defaults(hv.opts.Ellipse(line_width=3))
hv.opts.defaults(hv.opts.Area(fill_alpha=.5, line_width=1.5, line_color='gray', default_tools=[]))
hv.opts.defaults(hv.opts.Polygons(line_width=1.5, fill_alpha=0, line_alpha=0, line_color='gray', selection_line_color='black', hover_fill_alpha=.8, hover_line_alpha=1, selection_fill_alpha=.8, selection_line_alpha=1, nonselection_line_alpha=0, default_tools=[]))
hv.opts.defaults(hv.opts.Curve(line_width=5, line_color=BOARD_COLORS['ADC']))
hv.opts.defaults(hv.opts.Points(line_color=None, color=BOARD_COLORS['ADC'], size=15))


def panel_template(title, dark_mode):
    pn.config.theme = 'dark' if dark_mode else 'default'
    return pn.template.FastListTemplate(
        title=title,
        theme='dark' if dark_mode else 'default',
        theme_toggle=False,
        header_background='#262626',
        modal=[pn.Column()],
        collapsed_sidebar=True
    )


def text_color(dark_mode):
    # needed for pn.indicators.Number which doesn't respect pn.config.theme
    return 'white' if dark_mode else 'black'