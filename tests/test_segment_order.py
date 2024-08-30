from spinsight.main import get_segment_order

def test_segment_order():
    N = 5
    Nsym = 3
    assert get_segment_order(N, Nsym, c=0)==[3, 4, 2, 1, 0]
    assert get_segment_order(N, Nsym, c=1)==[4, 3, 2, 1, 0]
    assert get_segment_order(N, Nsym, c=2)==[1, 2, 3, 4, 0]
    assert get_segment_order(N, Nsym, c=3)==[0, 1, 2, 3, 4]
    assert get_segment_order(N, Nsym, c=4)==[0, 1, 2, 4, 3]
    Nsym = 2
    assert get_segment_order(N, Nsym, c=0)==[4, 3, 2, 1, 0]
    assert get_segment_order(N, Nsym, c=1)==[2, 3, 4, 1, 0]
    assert get_segment_order(N, Nsym, c=2)==[0, 1, 4, 3, 2]
    assert get_segment_order(N, Nsym, c=3)==[0, 1, 2, 3, 4]