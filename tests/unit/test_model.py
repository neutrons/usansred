# ===========================================================================
# Tests for XYData and IQData models
# ===========================================================================

from usansred.model import IQData, MonitorData, XYData


class TestXYData:
    """Tests for the XYData dataclass."""

    def test_default_construction(self):
        """Default XYData should have empty lists."""
        xy = XYData()
        assert xy.x == []
        assert xy.y == []
        assert xy.e == []
        assert xy.t == []

    def test_as_dict(self):
        """as_dict() should return a dictionary with X, Y, E, T keys."""
        xy = XYData(x=[1.0, 2.0], y=[3.0, 4.0], e=[0.1, 0.2], t=[10.0, 20.0])
        d = xy.as_dict()
        assert d == {"X": [1.0, 2.0], "Y": [3.0, 4.0], "E": [0.1, 0.2], "T": [10.0, 20.0]}

    def test_as_dict_empty(self):
        """as_dict() on empty XYData should return empty lists."""
        xy = XYData()
        d = xy.as_dict()
        assert d == {"X": [], "Y": [], "E": [], "T": []}


class TestIQData:
    """Tests for the IQData dataclass."""

    def test_default_construction(self):
        """Default IQData should have empty lists."""
        iq = IQData()
        assert iq.q == []
        assert iq.i == []
        assert iq.e == []
        assert iq.t == []

    def test_as_dict(self):
        """as_dict() should return a dictionary with Q, I, E, T keys."""
        iq = IQData(q=[0.1, 0.2], i=[100.0, 200.0], e=[10.0, 14.0], t=[1.0, 2.0])
        d = iq.as_dict()
        assert d == {"Q": [0.1, 0.2], "I": [100.0, 200.0], "E": [10.0, 14.0], "T": [1.0, 2.0]}

    def test_as_dict_empty(self):
        """as_dict() on empty IQData should return empty lists."""
        iq = IQData()
        d = iq.as_dict()
        assert d == {"Q": [], "I": [], "E": [], "T": []}


class TestMonitorData:
    """Tests for the MonitorData dataclass."""

    def test_default_construction(self):
        """Default MonitorData should have empty XYData, IQData, and empty filepath."""
        md = MonitorData()
        assert md.xy_data.x == []
        assert md.iq_data.q == []
        assert md.filepath == ""

    def test_construction_with_data(self):
        """MonitorData can be constructed with specific data."""
        xy = XYData(x=[1.0], y=[2.0], e=[0.1], t=[])
        iq = IQData(q=[1.0], i=[2.0], e=[0.1], t=[])
        md = MonitorData(xy_data=xy, iq_data=iq, filepath="/tmp/test.txt")
        assert md.xy_data.x == [1.0]
        assert md.iq_data.q == [1.0]
        assert md.filepath == "/tmp/test.txt"
