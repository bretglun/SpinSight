from spinsight.DAG import Graph
# Initialize graph node decorators:
from spinsight.nodes import (
    internal_input_params,
    helpers,
    param_bounds,
    setup_sequence_objects,
    sequence_timing,
    phase_encoding_order,
    place_sequence_objects,
    kspace_simulation,
    kspace_processing,
    image_reconstruction,
    sequence_plot,
    kspace_plot,
    image_plot,
    SNR_and_scantime,
    set_ui_param_visibility,
    set_ui_param_bounds,
    set_ui_params,
)


def make_graph(controller, dashboard):
    return Graph(controller, dashboard)