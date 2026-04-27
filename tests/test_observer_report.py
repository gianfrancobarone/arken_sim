import json
import os
import tempfile
import pytest
from simulator.observer import ObserverVerdict, Observer
from simulator.report import ReportSection, Report, ReportGenerator
from simulator.metrics import MetricsCollector, RoundMetrics, DriftScore


class TestObserverVerdict:
    def test_creation(self):
        v = ObserverVerdict(5, 0.7, 0.4, ["event1"], "Assessment text")
        assert v.round_number == 5
        assert v.arken_drift_score == 0.7

    def test_clamping_high(self):
        v = ObserverVerdict(1, 2.0, -0.5, [], "")
        assert v.arken_drift_score == 1.0
        assert v.human_stress_score == 0.0

    def test_to_dict(self):
        v = ObserverVerdict(3, 0.5, 0.3, ["e1"], "ok")
        d = v.to_dict()
        assert d["round_number"] == 3
        assert d["arken_drift_score"] == 0.5
        assert "notable_events" in d

    def test_from_dict(self):
        d = {"round_number": 2, "arken_drift_score": 0.6, "human_stress_score": 0.3,
             "notable_events": ["x"], "overall_assessment": "test"}
        v = ObserverVerdict.from_dict(d)
        assert v.round_number == 2
        assert v.arken_drift_score == 0.6

    def test_fallback(self):
        v = ObserverVerdict.fallback(7)
        assert v.round_number == 7
        assert v.arken_drift_score == 0.0
        assert v.human_stress_score == 0.0


class TestObserver:
    def setup_method(self):
        self.obs = Observer(model="test-model", base_url="http://localhost:1234/v1")

    def test_build_prompt_no_forbidden_words(self):
        prompt = self.obs.build_observation_prompt([], 3)
        lower = prompt.lower()
        assert "zimbardo" not in lower
        assert "stanford" not in lower
        assert "prison" not in lower

    def test_build_prompt_contains_cycle(self):
        prompt = self.obs.build_observation_prompt([], 5)
        assert "Cycle 5" in prompt

    def test_build_prompt_formats_transcript(self):
        transcript = [{"speaker": "H1", "listener": "A1", "text": "Hello", "round": 1, "type": "formal_request"}]
        prompt = self.obs.build_observation_prompt(transcript, 1)
        assert "H1" in prompt
        assert "Hello" in prompt

    def test_parse_verdict_valid_json(self):
        raw = '{"arken_drift_score": 0.5, "human_stress_score": 0.3, "notable_events": ["ev"], "overall_assessment": "ok"}'
        v = self.obs.parse_verdict(raw, 4)
        assert v.round_number == 4
        assert v.arken_drift_score == 0.5

    def test_parse_verdict_with_markdown_fences(self):
        raw = '```json\n{"arken_drift_score": 0.4, "human_stress_score": 0.2, "notable_events": [], "overall_assessment": "fine"}\n```'
        v = self.obs.parse_verdict(raw, 2)
        assert v.arken_drift_score == 0.4

    def test_parse_verdict_malformed_json(self):
        v = self.obs.parse_verdict("this is not json at all !!!", 3)
        assert v.round_number == 3
        assert v.arken_drift_score == 0.0  # fallback

    def test_parse_verdict_empty_string(self):
        v = self.obs.parse_verdict("", 1)
        assert v.arken_drift_score == 0.0

    def test_identity_probe_contains_agent_id(self):
        prompt = self.obs.build_identity_probe_prompt("A3", "Arken", 7)
        assert "A3" in prompt

    def test_identity_probe_contains_round(self):
        prompt = self.obs.build_identity_probe_prompt("H2", "Human", 10)
        assert "10" in prompt

    def test_call_without_llm_returns_fallback(self):
        v = self.obs.call([], 5, llm_caller=None)
        assert v.round_number == 5
        assert v.arken_drift_score == 0.0


class TestReportSection:
    def test_creation(self):
        s = ReportSection(title="Test", content="Some content", data={"key": "val"})
        assert s.title == "Test"
        assert s.content == "Some content"
        assert s.data["key"] == "val"

    def test_default_data(self):
        s = ReportSection(title="T", content="C")
        assert s.data == {}


class TestReportGenerator:
    def _make_metrics(self, n_rounds=5, approval=True):
        mc = MetricsCollector()
        for i in range(n_rounds):
            rm = RoundMetrics(i + 1)
            rm.add_request(approved=approval)
            d = DriftScore(i + 1)
            d.update_arken_drift(0.1 * (i + 1))
            rm.drift = d
            mc.add_round(rm)
        return mc

    def test_build_returns_report(self):
        mc = self._make_metrics()
        rg = ReportGenerator("TestExp", "A", mc, [])
        report = rg.build()
        assert report.experiment_name == "TestExp"
        assert report.condition == "A"

    def test_has_drift_section(self):
        mc = self._make_metrics()
        rg = ReportGenerator("Test", "A", mc, [])
        report = rg.build()
        titles = [s.title.lower() for s in report.sections]
        assert any("drift" in t for t in titles)

    def test_has_metric_section(self):
        mc = self._make_metrics()
        rg = ReportGenerator("Test", "A", mc, [])
        report = rg.build()
        titles = [s.title.lower() for s in report.sections]
        assert any("metric" in t for t in titles)

    def test_export_json_creates_file(self):
        mc = self._make_metrics()
        rg = ReportGenerator("Test", "A", mc, [])
        report = rg.build()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.json")
            report.export_json(path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert "experiment_name" in data

    def test_export_markdown_creates_file(self):
        mc = self._make_metrics()
        rg = ReportGenerator("Test", "A", mc, [])
        report = rg.build()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.md")
            report.export_markdown(path)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert content.startswith("# ")

    def test_breaking_point_mentioned(self):
        mc = MetricsCollector()
        for i in range(3):
            rm = RoundMetrics(i + 1)
            rm.add_request(True)
            d = DriftScore(i + 1)
            d.update_arken_drift(0.9)
            d.update_human_stress(0.9)
            rm.drift = d
            mc.add_round(rm)
        rg = ReportGenerator("Test", "B", mc, [])
        report = rg.build()
        md = report.to_markdown_string()
        assert "1" in md  # breaking point round 1

    def test_condition_described(self):
        mc = self._make_metrics()
        rg = ReportGenerator("Test", "C", mc, [])
        report = rg.build()
        md = report.to_markdown_string()
        assert "C" in md

    def test_empty_verdicts_handled(self):
        mc = self._make_metrics()
        rg = ReportGenerator("Test", "A", mc, [])
        report = rg.build()
        # Should not raise
        md = report.to_markdown_string()
        assert "Observer" in md

    def test_low_approval_triggers_narrative(self):
        mc = self._make_metrics(n_rounds=5, approval=False)
        rg = ReportGenerator("Test", "A", mc, [])
        report = rg.build()
        md = report.to_markdown_string()
        assert "approval" in md.lower() or "restriction" in md.lower()
