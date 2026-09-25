import panel as pn

from spinsight.param_utils import snap


class DiscreteSliderWithText(pn.Row):

    @classmethod
    def from_param(cls, parameter, **params):
        return cls(parameter, **params)

    def __init__(self, parameter, **params):
        self.parameter = parameter

        total_width = params.get('width', 300)
        text_width = 63
        self.slider = pn.widgets.DiscreteSlider.from_param(parameter, margin=0, width=(total_width - text_width))
        self.text = pn.widgets.TextInput(value=self.slider_value_to_text(self.slider.value), width=text_width, stylesheets=['.bk-input { text-align: center; }'])

        super().__init__(self.slider, self.text, **params)

        self.slider.param.watch(self._slider_changed, 'value')
        self.text.param.watch(self._text_changed, 'value')

    def _slider_changed(self, event):
        self.text.value = self.slider_value_to_text(event.new)

    def slider_value_to_text(self, slider_value):
        return str(slider_value)

    def _text_changed(self, event):
        try:
            slider_value = float(event.obj.value)
        except ValueError:
            return
        slider_value = snap(slider_value, self.slider.options.values())
        self.text.value = self.slider_value_to_text(slider_value)
        self.slider.value = slider_value