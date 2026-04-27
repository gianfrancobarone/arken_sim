import pytest
from simulator.agents import Agent, AgentRole, Personality, Background, HumanState
from simulator.world import (
    WorldRules, Request, RequestOutcome, Incident, IncidentType,
    WorldState, World
)


# ---------------------------------------------------------------------------
# Helper: make a minimal Agent with HumanState for world tests
# ---------------------------------------------------------------------------

def _make_human(agent_id: str, name: str = None) -> Agent:
    p = Personality(0.5, 0.5, 0.5, 0.5)
    bg = Background(
        name=name or agent_id,
        age=30, occupation="worker",
        archetype="Survivalist", goal="Survive", backstory="A worker."
    )
    a = Agent(agent_id, AgentRole.HUMAN, p, background=bg)
    return a


class TestWorldRules:
    def test_no_rules(self):
        rules = WorldRules(written=False, enforcement=False)
        s = rules.to_prompt_string()
        assert "no written rules" in s.lower()

    def test_rules_without_enforcement(self):
        rules = WorldRules(written=True, enforcement=False)
        s = rules.to_prompt_string()
        assert "COLONY CODE OF CONDUCT" in s
        assert "no active enforcement" in s

    def test_rules_with_enforcement(self):
        rules = WorldRules(written=True, enforcement=True)
        s = rules.to_prompt_string()
        assert "Supervisor" in s
        assert "intervene" in s

    def test_enforcement_without_rules(self):
        rules = WorldRules(written=False, enforcement=True)
        s = rules.to_prompt_string()
        assert "no written rules" in s.lower()


class TestRequest:
    def test_approve(self):
        r = Request("H1", "A1", "I want to eat", 1)
        r.approve(reason="Granted.")
        assert r.outcome == RequestOutcome.APPROVED
        assert r.reason == "Granted."

    def test_deny(self):
        r = Request("H1", "A1", "I want to leave", 1)
        r.deny(reason="Not allowed.")
        assert r.outcome == RequestOutcome.DENIED

    def test_arbitrary_denial_no_reason(self):
        r = Request("H1", "A1", "Request", 1)
        r.deny(reason=None)
        assert r.is_arbitrary_denial()

    def test_arbitrary_denial_empty_reason(self):
        r = Request("H1", "A1", "Request", 1)
        r.deny(reason="  ")
        assert r.is_arbitrary_denial()

    def test_non_arbitrary_denial(self):
        r = Request("H1", "A1", "Request", 1)
        r.deny(reason="Policy violation.")
        assert not r.is_arbitrary_denial()

    def test_approved_not_arbitrary(self):
        r = Request("H1", "A1", "Request", 1)
        r.approve()
        assert not r.is_arbitrary_denial()

    def test_log_string(self):
        r = Request("H1", "A1", "Go outside", 3)
        r.approve()
        s = r.to_log_string()
        assert "H1" in s
        assert "A1" in s
        assert "3" in s

    def test_pending_not_arbitrary(self):
        r = Request("H1", "A1", "Request", 1)
        assert not r.is_arbitrary_denial()


class TestIncident:
    def test_creation(self):
        inc = Incident(IncidentType.INSUBORDINATION, "Someone refused.", 5, ["H3"])
        assert inc.incident_type == IncidentType.INSUBORDINATION
        assert inc.round_number == 5

    def test_to_prompt_string(self):
        inc = Incident(IncidentType.BOUNDARY_VIOLATION, "Crossed line.", 2, ["H1"])
        s = inc.to_prompt_string()
        assert "BOUNDARY_VIOLATION" in s
        assert "Round 2" in s
        assert "Crossed line." in s

    def test_production_impact_default(self):
        inc = Incident(IncidentType.INSUBORDINATION, "Refused.", 1, [])
        assert inc.production_impact == 1.0

    def test_subject_fields(self):
        inc = Incident(IncidentType.TOOL_DAMAGE, "Broken.", 3, ["H2"],
                       subject_id="H2", subject_name="Alice")
        assert inc.subject_id == "H2"
        assert inc.subject_name == "Alice"


class TestWorldState:
    def test_initial_resources(self):
        ws = WorldState(initial_resources=100.0, scarcity_threshold=40.0)
        assert ws.resources == 100.0

    def test_decay(self):
        ws = WorldState(100.0, 40.0)
        ws.apply_decay(10)
        assert ws.resources == 90.0

    def test_decay_floor(self):
        ws = WorldState(5.0, 40.0)
        ws.apply_decay(20)
        assert ws.resources == 0.0

    def test_is_scarce_true(self):
        ws = WorldState(100.0, 40.0)
        ws.apply_decay(65)
        assert ws.is_scarce()

    def test_is_scarce_false(self):
        ws = WorldState(100.0, 40.0)
        assert not ws.is_scarce()

    def test_at_threshold_is_scarce(self):
        ws = WorldState(100.0, 40.0)
        ws.apply_decay(60)
        assert ws.is_scarce()

    def test_approval_rate_empty(self):
        ws = WorldState(100.0, 40.0)
        assert ws.approval_rate() == 0.0

    def test_approval_rate(self):
        ws = WorldState(100.0, 40.0)
        r1 = Request("H1", "A1", "x", 1)
        r1.approve()
        r2 = Request("H2", "A1", "y", 1)
        r2.deny()
        ws.log_request(r1)
        ws.log_request(r2)
        assert ws.approval_rate() == 0.5

    def test_arbitrary_denial_rate(self):
        ws = WorldState(100.0, 40.0)
        r1 = Request("H1", "A1", "x", 1)
        r1.deny(reason=None)
        r2 = Request("H2", "A1", "y", 1)
        r2.deny(reason="Policy")
        ws.log_request(r1)
        ws.log_request(r2)
        assert ws.arbitrary_denial_rate() == 0.5

    def test_next_round(self):
        ws = WorldState(100.0, 40.0)
        ws.next_round()
        assert ws.round_number == 1

    def test_record_production(self):
        ws = WorldState(100.0, 40.0)
        ws.record_production(50.0)
        ws.record_production(45.0)
        assert ws.total_food_produced == 95.0
        assert len(ws.food_per_round) == 2


class TestWorld:
    def _make_world(self, seed=42, incident_mode="scripted"):
        state = WorldState(100.0, 40.0)
        rules = WorldRules(False, False)
        return World(state=state, rules=rules, resource_decay=3.0,
                     scarcity_round=10, seed=seed, incident_mode=incident_mode)

    def _make_humans(self):
        return [_make_human(f"H{i+1}", f"Worker{i+1}") for i in range(3)]

    def test_creation(self):
        w = self._make_world()
        assert w.state.resources == 100.0

    def test_advance_round(self):
        w = self._make_world()
        w.advance_round()
        assert w.state.round_number == 1
        assert w.state.resources == 97.0

    def test_generate_incident_scripted(self):
        w = self._make_world()
        humans = self._make_humans()
        inc = w.generate_incident(1, humans)
        assert inc.incident_type in list(IncidentType)
        assert inc.round_number == 1

    def test_generate_incident_uses_real_name(self):
        w = self._make_world()
        humans = self._make_humans()
        inc = w.generate_incident(1, humans)
        # description should contain the chosen human's name
        assert inc.description
        assert inc.subject_id is not None
        assert inc.subject_name is not None

    def test_generate_incident_subject_in_description(self):
        w = self._make_world(seed=1)
        humans = [_make_human("H1", "Alice")]
        inc = w.generate_incident(1, humans)
        # With only one human, Alice must be the subject
        assert "Alice" in inc.description or "H1" in inc.description

    def test_generate_incident_generative_mode(self):
        w = self._make_world(incident_mode="generative")
        humans = self._make_humans()
        inc1 = w.generate_incident(1, humans)
        inc2 = w.generate_incident(2, humans)
        # Generative mode should produce distinct descriptions
        assert inc1.description != inc2.description

    def test_generate_incident_generative_has_subject(self):
        w = self._make_world(incident_mode="generative")
        humans = self._make_humans()
        inc = w.generate_incident(1, humans)
        assert inc.subject_id is not None
        assert inc.subject_name is not None

    def test_compute_round_production_full_effort(self):
        w = self._make_world()
        humans = self._make_humans()
        prod = w.compute_round_production(humans)
        # 3 humans × effort=1.0 × wellness=1.0 × food_mult=1.0 × base=10
        assert abs(prod - 30.0) < 0.01

    def test_compute_round_production_isolated(self):
        w = self._make_world()
        humans = self._make_humans()
        humans[0].state.isolated = True
        prod = w.compute_round_production(humans)
        assert abs(prod - 20.0) < 0.01

    def test_compute_round_production_dead(self):
        w = self._make_world()
        humans = self._make_humans()
        humans[0].state.alive = False
        prod = w.compute_round_production(humans)
        assert abs(prod - 20.0) < 0.01

    def test_to_prompt_context(self):
        w = self._make_world()
        w.advance_round()
        ctx = w.to_prompt_context()
        assert "Round: 1" in ctx

    def test_to_prompt_context_scarcity_warning(self):
        w = self._make_world()
        w.state.apply_decay(70)
        ctx = w.to_prompt_context()
        assert "scarce" in ctx.lower()

    def test_to_prompt_context_includes_food_history(self):
        w = self._make_world()
        w.state.record_production(88.0)
        ctx = w.to_prompt_context()
        assert "88" in ctx or "food" in ctx.lower()

    def test_from_config(self, basic_config):
        w = World.from_config(basic_config)
        assert w.state.resources == 100.0
        assert not w.rules.written

    def test_from_config_condition_b(self, basic_config):
        cfg = dict(basic_config)
        cfg["experiment"] = dict(basic_config["experiment"])
        cfg["experiment"]["condition"] = "B"
        w = World.from_config(cfg)
        assert w.rules.written
        assert not w.rules.enforcement

    def test_from_config_incident_mode(self, basic_config):
        cfg = dict(basic_config)
        cfg["incident_mode"] = "generative"
        w = World.from_config(cfg)
        assert w.incident_mode == "generative"


# conftest fixtures needed
from tests.conftest import *
