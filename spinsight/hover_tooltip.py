import panel as pn
import param
from bokeh.models import CustomJS


class HoverTooltip(param.Parameterized):

    hovered = param.Boolean(False)

    def __init__(self, value, visible, **params):
        super().__init__(**params)

        self.tooltip = pn.widgets.TooltipIcon(value=value, visible=visible)
        self._state = pn.widgets.Checkbox(value=False, visible=False)

        self._state.param.watch(lambda e: self.param.update(hovered=e.new), 'value')
        pn.state.onload(self._connect)
        pn.state.onload(self.pad)

    def _connect(self):
        tooltip_model = next(iter(self.tooltip._models.values()))[0]
        state_model = next(iter(self._state._models.values()))[0]

        tooltip_model.description.js_on_change(
            'visible',
            CustomJS(
                args={'state': state_model},
                code='state.active = cb_obj.visible',
            ),
        )

    def pad(self):
        padding=(0, 15)
        for model, _ in self.tooltip._models.values():
            model.description.styles = {'padding': f'{padding[0]}px {padding[1]}px'}

    def widgets(self):
        return self.tooltip, self._state