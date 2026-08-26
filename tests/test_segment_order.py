from spinsight.nodes.phase_encoding_order import flexible_segment_order


def test_segment_order():
    num_segm = 5
    num_sym = 3
    assert flexible_segment_order(num_segm, num_sym, k0_index=0)==[3, 4, 2, 1, 0]
    assert flexible_segment_order(num_segm, num_sym, k0_index=1)==[4, 3, 2, 1, 0]
    assert flexible_segment_order(num_segm, num_sym, k0_index=2)==[1, 2, 3, 4, 0]
    assert flexible_segment_order(num_segm, num_sym, k0_index=3)==[0, 1, 2, 3, 4]
    assert flexible_segment_order(num_segm, num_sym, k0_index=4)==[0, 1, 2, 4, 3]
    num_sym = 2
    assert flexible_segment_order(num_segm, num_sym, k0_index=0)==[4, 3, 2, 1, 0]
    assert flexible_segment_order(num_segm, num_sym, k0_index=1)==[4, 3, 2, 1, 0]
    assert flexible_segment_order(num_segm, num_sym, k0_index=2)==[2, 3, 4, 1, 0]
    assert flexible_segment_order(num_segm, num_sym, k0_index=3)==[0, 1, 2, 3, 4]
    assert flexible_segment_order(num_segm, num_sym, k0_index=4)==[0, 1, 2, 3, 4]