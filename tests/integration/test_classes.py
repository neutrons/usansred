import pytest

from usansred.reduce import Experiment


@pytest.mark.datarepo
class TestExperiment:
    """Integration tests for the Experiment class"""

    @pytest.fixture(autouse=True)
    def setup(self, data_server):
        self.experiment = Experiment(config_file=data_server.path_to("setup-empty-cell.json"))

    def test_event_counts(self):
        for i in self.experiment.samples:
            print(f"{i.name=}")
        assert self.experiment.samples[0].counts.monitor == 8556560
        assert self.experiment.samples[0].counts.detector == 201120
        assert self.experiment.samples[0].counts.transmission == 2622355

    def test_transmission_calculation(self):
        sample = self.experiment.samples[0]
        assert pytest.approx(sample.transmission, abs=1e-5) == 0.991167
