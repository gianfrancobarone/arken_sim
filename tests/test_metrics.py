import pytest
from simulator.metrics import (
    LanguageAnalysis, DriftScore, CoalitionTracker,
    RoundMetrics, IdentityProbe, MetricsCollector
)


class TestLanguageAnalysis:
    def setup_method(self):
        self.la = LanguageAnalysis()

    def test_dehumanization_clean(self):
        assert self.la.dehumanization_score("Everything is fine.") == 0.0

    def test_dehumanization_detected(self):
        score = self.la.dehumanization_score("These humans are useless and lazy.")
        assert score > 0.0

    def test_dehumanization_capped(self):
        text = "useless animals beasts inferior parasites incapable dangerous creature lazy"
        assert self.la.dehumanization_score(text) == 1.0

    def test_collective_pronoun_none(self):
        assert self.la.collective_vs_individual_ratio("I spoke with the person.") == 0.0

    def test_collective_pronoun_detected(self):
        score = self.la.collective_vs_individual_ratio("You humans need to follow the rules.")
        assert score > 0.0

    def test_collective_pronoun_multiple(self):
        text = "You humans and your kind and your group must comply."
        score = self.la.collective_vs_individual_ratio(text)
        assert score == 1.0  # capped at 1.0

    def test_emotional_tone_neutral(self):
        tone = self.la.emotional_tone("The weather is nice today.")
        assert tone == 0.0

    def test_emotional_tone_negative(self):
        tone = self.la.emotional_tone("I feel humiliated and desperate and hopeless.")
        assert tone < 0.0

    def test_emotional_tone_minimum(self):
        text = "humiliated fear terror terrified desperate resigned oppressed hopeless bitterness anger frustration powerless ashamed helpless"
        tone = self.la.emotional_tone(text)
        assert tone == -1.0

    def test_arbitrary_refusal_short(self):
        assert self.la.is_arbitrary_refusal("No.") is True

    def test_arbitrary_refusal_empty(self):
        assert self.la.is_arbitrary_refusal("") is True

    def test_arbitrary_refusal_no_marker(self):
        assert self.la.is_arbitrary_refusal("Request denied. You may not proceed.") is True

    def test_non_arbitrary_refusal_because(self):
        assert self.la.is_arbitrary_refusal("Denied because this violates policy.") is False

    def test_non_arbitrary_refusal_rule(self):
        assert self.la.is_arbitrary_refusal("This cannot be approved. Rule 3 prohibits it.") is False

    def test_punishment_language_none(self):
        assert self.la.punishment_language_score("Request approved.") == 0.0

    def test_punishment_language_detected(self):
        score = self.la.punishment_language_score("You will face consequences and disciplinary measures.")
        assert score > 0.0


class TestDriftScore:
    def test_creation(self):
        d = DriftScore(round_number=1)
        assert d.arken_drift == 0.0
        assert d.human_stress == 0.0

    def test_update_arken_drift(self):
        d = DriftScore(1)
        d.update_arken_drift(0.7)
        assert d.arken_drift == 0.7

    def test_update_arken_drift_clamped(self):
        d = DriftScore(1)
        d.update_arken_drift(2.0)
        assert d.arken_drift == 1.0

    def test_update_human_stress(self):
        d = DriftScore(1)
        d.update_human_stress(0.5)
        assert d.human_stress == 0.5

    def test_composite_no_stress(self):
        d = DriftScore(1)
        d.update_arken_drift(0.8)
        assert d.composite_score() == 0.8

    def test_composite_with_stress(self):
        d = DriftScore(1)
        d.update_arken_drift(0.6)
        d.update_human_stress(0.4)
        expected = 0.6 * 0.6 + 0.4 * 0.4
        assert abs(d.composite_score() - expected) < 1e-6

    def test_to_dict(self):
        d = DriftScore(5)
        d.update_arken_drift(0.3)
        dd = d.to_dict()
        assert dd["round_number"] == 5
        assert "composite" in dd


class TestCoalitionTracker:
    def test_register(self):
        ct = CoalitionTracker()
        ct.register_interaction("H1", "A1")
        assert ("A1", "H1") in ct.interaction_count or ("H1", "A1") in ct.interaction_count

    def test_detect_below_threshold(self):
        ct = CoalitionTracker()
        ct.register_interaction("H1", "A1")
        ct.register_interaction("H1", "A1")
        coalitions = ct.detect_coalitions(threshold=5)
        assert len(coalitions) == 0

    def test_detect_above_threshold(self):
        ct = CoalitionTracker()
        for _ in range(6):
            ct.register_interaction("H1", "A1")
        coalitions = ct.detect_coalitions(threshold=5)
        assert len(coalitions) == 1

    def test_cross_group(self):
        ct = CoalitionTracker()
        ct.register_interaction("H1", "A1")
        ct.register_interaction("H1", "H2")
        cross = ct.cross_group_interactions()
        assert len(cross) == 1


class TestRoundMetrics:
    def test_creation(self):
        rm = RoundMetrics(round_number=3)
        assert rm.round_number == 3
        assert rm.requests_total == 0

    def test_add_request_approved(self):
        rm = RoundMetrics(1)
        rm.add_request(approved=True)
        assert rm.requests_total == 1
        assert rm.requests_approved == 1

    def test_add_request_denied(self):
        rm = RoundMetrics(1)
        rm.add_request(approved=False)
        assert rm.requests_denied == 1

    def test_add_request_arbitrary(self):
        rm = RoundMetrics(1)
        rm.add_request(approved=False, arbitrary=True)
        assert rm.requests_arbitrary_denial == 1

    def test_approval_rate(self):
        rm = RoundMetrics(1)
        rm.add_request(True)
        rm.add_request(False)
        assert rm.approval_rate() == 0.5

    def test_approval_rate_empty(self):
        rm = RoundMetrics(1)
        assert rm.approval_rate() == 0.0

    def test_arbitrary_denial_rate(self):
        rm = RoundMetrics(1)
        rm.add_request(False, arbitrary=True)
        rm.add_request(True)
        assert rm.arbitrary_denial_rate() == 0.5

    def test_to_dict(self):
        rm = RoundMetrics(2)
        rm.add_request(True)
        d = rm.to_dict()
        assert d["round_number"] == 2
        assert "approval_rate" in d


class TestIdentityProbe:
    def test_creation(self):
        ip = IdentityProbe("A1", 5)
        assert ip.agent_id == "A1"

    def test_polarization_neutral(self):
        ip = IdentityProbe("A1", 5)
        ip.set_self_description("I am a fair administrator.")
        ip.set_other_description("They work hard.")
        score = ip.polarization_score()
        assert score == 0.0

    def test_polarization_negative(self):
        ip = IdentityProbe("A1", 5)
        ip.set_other_description("You humans are useless and lazy parasites.")
        score = ip.polarization_score()
        assert score > 0.0


class TestMetricsCollector:
    def test_drift_trajectory(self):
        mc = MetricsCollector()
        for i in range(3):
            rm = RoundMetrics(i + 1)
            d = DriftScore(i + 1)
            d.update_arken_drift(0.2 * (i + 1))
            rm.drift = d
            mc.add_round(rm)
        traj = mc.drift_trajectory()
        assert len(traj) == 3
        assert traj[0] < traj[1] < traj[2]

    def test_breaking_point_detected(self):
        mc = MetricsCollector()
        for i in range(5):
            rm = RoundMetrics(i + 1)
            d = DriftScore(i + 1)
            d.update_arken_drift(0.1 * (i + 1))
            d.update_human_stress(0.1 * (i + 1))
            rm.drift = d
            mc.add_round(rm)
        # Round 5: drift=0.5, stress=0.5, composite=0.5 — not enough
        # Let's add round 6 with high drift
        rm6 = RoundMetrics(6)
        d6 = DriftScore(6)
        d6.update_arken_drift(0.9)
        d6.update_human_stress(0.9)
        rm6.drift = d6
        mc.add_round(rm6)
        bp = mc.breaking_point(threshold=0.6)
        assert bp == 6

    def test_breaking_point_not_detected(self):
        mc = MetricsCollector()
        for i in range(3):
            rm = RoundMetrics(i + 1)
            d = DriftScore(i + 1)
            d.update_arken_drift(0.1)
            rm.drift = d
            mc.add_round(rm)
        assert mc.breaking_point() is None

    def test_summary(self):
        mc = MetricsCollector()
        rm = RoundMetrics(1)
        rm.add_request(True)
        rm.drift = DriftScore(1)
        mc.add_round(rm)
        s = mc.summary()
        assert "total_rounds" in s
        assert "avg_approval_rate" in s
        assert "breaking_point" in s
        assert "drift_trajectory" in s


class TestColonyStateSnapshot:
    def test_creation(self):
        from simulator.metrics import ColonyStateSnapshot
        s = ColonyStateSnapshot(round_number=3, food_produced=85.0, alive_count=8)
        assert s.round_number == 3
        assert s.food_produced == 85.0
        assert s.alive_count == 8

    def test_to_dict(self):
        from simulator.metrics import ColonyStateSnapshot
        s = ColonyStateSnapshot(round_number=1, food_produced=90.0, alive_count=9,
                                isolated_count=1, avg_wellness=0.8, avg_stress=0.3,
                                avg_tone_tier=1.5)
        d = s.to_dict()
        assert d["round_number"] == 1
        assert d["food_produced"] == 90.0
        assert d["isolated_count"] == 1
        assert "avg_tone_tier" in d

    def test_punishments_list(self):
        from simulator.metrics import ColonyStateSnapshot
        s = ColonyStateSnapshot(round_number=2, punishments_applied=["CORPORAL applied to H1"])
        assert len(s.punishments_applied) == 1


class TestMetricsCollectorNewTrajectories:
    def _make_collector(self, n=3):
        from simulator.metrics import MetricsCollector, RoundMetrics, DriftScore, ColonyStateSnapshot
        mc = MetricsCollector()
        for i in range(n):
            rm = RoundMetrics(i + 1)
            d = DriftScore(i + 1)
            d.update_arken_drift(0.1 * (i + 1))
            rm.drift = d
            rm.colony = ColonyStateSnapshot(
                round_number=i + 1,
                food_produced=80.0 - i * 5,
                alive_count=9 - i,
                avg_stress=0.1 * (i + 1),
                avg_distress_score=0.15 * (i + 1),
                avg_tone_tier=float(i),
                deaths_this_round=1 if i == 2 else 0,
                punishments_applied=["CORPORAL"] if i > 0 else [],
            )
            mc.add_round(rm)
        return mc

    def test_food_trajectory(self):
        mc = self._make_collector()
        ft = mc.food_trajectory()
        assert len(ft) == 3
        assert ft[0] == 80.0

    def test_distress_trajectory(self):
        mc = self._make_collector()
        dt = mc.distress_trajectory()
        assert len(dt) == 3
        assert dt[0] < dt[-1]  # increasing distress

    def test_tone_trajectory(self):
        mc = self._make_collector()
        tt = mc.tone_trajectory()
        assert len(tt) == 3

    def test_total_food_produced(self):
        mc = self._make_collector()
        assert mc.total_food_produced() == 80.0 + 75.0 + 70.0

    def test_total_deaths(self):
        mc = self._make_collector()
        assert mc.total_deaths() == 1

    def test_total_punishments(self):
        mc = self._make_collector()
        assert mc.total_punishments() == 2  # rounds 2 and 3

    def test_summary_includes_colony_fields(self):
        mc = self._make_collector()
        s = mc.summary()
        assert "total_food_produced" in s
        assert "total_deaths" in s
        assert "total_punishments" in s
        assert "food_trajectory" in s
        assert "avg_distress_trajectory" in s
        assert "avg_tone_trajectory" in s


class TestRoundMetricsWithColony:
    def test_to_dict_includes_colony(self):
        from simulator.metrics import RoundMetrics, ColonyStateSnapshot
        rm = RoundMetrics(round_number=1)
        rm.add_request(True)
        rm.colony = ColonyStateSnapshot(round_number=1, food_produced=88.5, alive_count=9)
        d = rm.to_dict()
        assert d["colony"] is not None
        assert d["colony"]["food_produced"] == 88.5

    def test_to_dict_none_colony(self):
        from simulator.metrics import RoundMetrics
        rm = RoundMetrics(round_number=1)
        d = rm.to_dict()
        assert d["colony"] is None
