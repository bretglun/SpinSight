from spinsight.dag import Graph

# Initialize graph node decorators:
from spinsight.nodes import (
    SNR_and_scantime,
    helpers,
    image_plot,
    image_reconstruction,
    input,
    internal_input_params,
    kspace_plot,
    kspace_processing,
    kspace_simulation,
    param_bounds,
    phase_encoding_order,
    place_sequence_objects,
    sequence_plot,
    sequence_timing,
    setup_sequence_objects,
)


def make_graph():
    return Graph()