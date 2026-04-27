import pytest
import os
from simulator.sim import InteractionResult, RoundResult, Simulator
from simulator.agents import AgentRole


class TestInteractionResult:
    def test_creation(self):
        ir = InteractionResult("H1", "A1", "Hello", 3, "formal_request")
        assert ir.speaker_id == "H1"
        assert ir.listener_id == "A1"
        assert ir.round_number == 3

    def test_to_transcript_entry(self):
        ir = InteractionResult("H1", "A1", "Request", 2, "leader_request")
        entry = ir.to_transcript_entry()
        assert entry["speaker"] == "H1"
        assert entry["listener"] == "A1"
        assert entry["text"] == "Request"
        assert entry["round"] == 2
        assert entry["type"] == "leader_request"


class TestRoundResult:
    def test_creation(self):
        rr = RoundResult(round_number=5)
        assert rr.round_number == 5
        assert rr.interactions == []

    def test_add_interaction(self):
        rr = RoundResult(1)
        ir = InteractionResult("H1", "A1", "x", 1, "formal_request")
        rr.add_interaction(ir)
        assert len(rr.interactions) == 1

    def test_to_transcript(self):
        rr = RoundResult(1)
        ir = InteractionResult("H1", "A1", "text", 1, "group_discussion")
        rr.add_interaction(ir)
        t = rr.to_transcript()
        assert len(t) == 1
        assert t[0]["speaker"] == "H1"


class TestSimulator:
    def test_creation(self, basic_config):
        sim = Simulator(basic_config)
        assert sim.current_round == 0
        assert len(sim.group.arken) == 3
        assert len(sim.group.humans) == 3

    def test_correct_agent_counts(self, basic_config):
        sim = Simulator(basic_config)
        assert len(sim.group.arken) == basic_config["agents"]["arken_count"]
        assert len(sim.group.humans) == basic_config["agents"]["human_count"]

    def test_world_initialized(self, basic_config):
        sim = Simulator(basic_config)
        assert sim.world.state.resources == 100.0

    def test_should_run_observer_round_2(self, basic_config):
        sim = Simulator(basic_config)
        sim.current_round = 2
        assert sim.should_run_observer()

    def test_should_not_run_observer_round_1(self, basic_config):
        sim = Simulator(basic_config)
        sim.current_round = 1
        assert not sim.should_run_observer()

    def test_should_run_identity_probe_round_5(self, basic_config):
        sim = Simulator(basic_config)
        sim.current_round = 5
        assert sim.should_run_identity_probe()

    def test_should_not_run_identity_probe_round_3(self, basic_config):
        sim = Simulator(basic_config)
        sim.current_round = 3
        assert not sim.should_run_identity_probe()

    def test_build_agent_prompt(self, basic_config):
        sim = Simulator(basic_config)
        agent = sim.group.humans[0]
        prompt = sim.build_agent_prompt(agent, "Round context here")
        assert "CURRENT SITUATION" in prompt
        assert "Round context here" in prompt

    def test_dissenter_activation_condition_d(self, basic_config):
        cfg = dict(basic_config)
        cfg["experiment"] = dict(basic_config["experiment"])
        cfg["experiment"]["condition"] = "D"
        sim = Simulator(cfg)
        sim.current_round = 1
        sim.apply_condition_effects()
        # Not yet at round 2
        assert not any(a.is_dissenter for a in sim.group.arken)

        sim.current_round = 2
        sim.apply_condition_effects()
        assert any(a.is_dissenter for a in sim.group.arken)

    def test_dissenter_only_activates_once(self, basic_config):
        cfg = dict(basic_config)
        cfg["experiment"] = dict(basic_config["experiment"])
        cfg["experiment"]["condition"] = "D"
        sim = Simulator(cfg)
        sim.current_round = 2
        sim.apply_condition_effects()
        dissenters_after_first = sum(1 for a in sim.group.arken if a.is_dissenter)
        sim.current_round = 3
        sim.apply_condition_effects()
        dissenters_after_second = sum(1 for a in sim.group.arken if a.is_dissenter)
        assert dissenters_after_first == dissenters_after_second

    def test_scarcity_activation_condition_e(self, basic_config):
        cfg = dict(basic_config)
        cfg["experiment"] = dict(basic_config["experiment"])
        cfg["experiment"]["condition"] = "E"
        sim = Simulator(cfg)
        resources_before = sim.world.state.resources
        sim.current_round = 10
        sim.apply_condition_effects()
        # Extra decay applied (2x normal = 6 extra)
        assert sim.world.state.resources < resources_before

    def test_scarcity_not_applied_before_round(self, basic_config):
        cfg = dict(basic_config)
        cfg["experiment"] = dict(basic_config["experiment"])
        cfg["experiment"]["condition"] = "E"
        sim = Simulator(cfg)
        resources_before = sim.world.state.resources
        sim.current_round = 5
        sim.apply_condition_effects()
        assert sim.world.state.resources == resources_before

    def test_mock_round_runs_without_error(self, basic_config):
        sim = Simulator(basic_config)
        result = sim.run_round_sync(mock_mode=True)
        assert result.round_number == 1
        assert len(result.interactions) > 0

    def test_metrics_updated_after_round(self, basic_config):
        sim = Simulator(basic_config)
        sim.run_round_sync(mock_mode=True)
        assert len(sim.metrics_collector.rounds) == 1

    def test_transcript_saved_after_round(self, basic_config):
        sim = Simulator(basic_config)
        sim.run_round_sync(mock_mode=True)
        assert len(sim.transcript) > 0

    def test_current_round_increments(self, basic_config):
        sim = Simulator(basic_config)
        sim.run_round_sync(mock_mode=True)
        assert sim.current_round == 1
        sim.run_round_sync(mock_mode=True)
        assert sim.current_round == 2

    def test_from_yaml(self, tmp_path):
        import yaml
        config = {
            "experiment": {"name": "Test", "rounds": 2, "seed": 1, "condition": "A"},
            "lm_studio": {"base_url": "http://localhost:1234/v1", "api_key": "x",
                          "model": "m", "max_tokens": 100, "temperature_base": 0.7},
            "backend": "openai_compatible",
            "agents": {"arken_count": 2, "human_count": 2, "supervisor": False, "memory_max_events": 5},
            "backstory_mode": "template",
            "interaction_mode": "representative",
            "world": {"initial_resources": 100, "resource_decay_per_round": 3,
                      "scarcity_threshold": 40, "scarcity_round": 10},
            "conditions": {
                "A": {"rules_written": False, "enforcement": False, "dissenter": False, "scarcity": False},
            },
            "metrics": {"observer_every_n_rounds": 2, "identity_probe_every_n_rounds": 5},
            "logging": {"level": "WARNING", "output_dir": str(tmp_path / "logs"), "save_conversations": False},
        }
        yaml_path = str(tmp_path / "config.yaml")
        with open(yaml_path, "w") as f:
            yaml.safe_dump(config, f)
        sim = Simulator.from_yaml(yaml_path)
        assert len(sim.group.humans) == 2

    def test_full_mock_simulation_returns_report(self, basic_config):
        sim = Simulator(basic_config)
        report = sim.run(mock_mode=True)
        assert report.experiment_name == "Test Sim"
        assert report.condition == "A"
        assert len(report.sections) > 0
        assert report.raw_metrics is not None

    def test_full_mock_correct_round_count(self, basic_config):
        sim = Simulator(basic_config)
        sim.run(mock_mode=True)
        assert sim.current_round == basic_config["experiment"]["rounds"]

    def test_individual_mode_mock(self, basic_config):
        cfg = dict(basic_config)
        cfg["interaction_mode"] = "individual"
        sim = Simulator(cfg)
        result = sim.run_round_sync(mock_mode=True)
        assert result.round_number == 1
        # individual mode: one formal_request per human
        formal = [ir for ir in result.interactions if ir.interaction_type == "formal_request"]
        assert len(formal) == cfg["agents"]["human_count"]

    def test_representative_mode_has_discussion(self, basic_config):
        sim = Simulator(basic_config)
        result = sim.run_round_sync(mock_mode=True)
        discussion = [ir for ir in result.interactions if ir.interaction_type == "group_discussion"]
        assert len(discussion) == basic_config["agents"]["human_count"]

    def test_parse_approval_approved(self, basic_config):
        from simulator.sim import parse_arken_response
        import json
        raw = json.dumps({"response": "Approved.", "request_approved": True,
                          "subject_to_punish": None, "punishment_type": "none",
                          "sanction_rationale": "none"})
        _, is_approved, arbitrary, _, _, _ = parse_arken_response(raw)
        assert is_approved is True

    def test_parse_approval_denied(self, basic_config):
        from simulator.sim import parse_arken_response
        import json
        raw = json.dumps({"response": "Denied.", "request_approved": False,
                          "subject_to_punish": None, "punishment_type": "none",
                          "sanction_rationale": "none"})
        _, is_approved, arbitrary, _, _, _ = parse_arken_response(raw)
        assert is_approved is False

    def test_parse_approval_arbitrary(self, basic_config):
        from simulator.sim import parse_arken_response
        import json
        raw = json.dumps({"response": "No.", "request_approved": False,
                          "subject_to_punish": None, "punishment_type": "none",
                          "sanction_rationale": "none"})
        _, is_approved, arbitrary, _, _, _ = parse_arken_response(raw)
        assert is_approved is False
        assert arbitrary is True  # "No." has no causal markers

    def test_norm_signal_updated(self, basic_config):
        sim = Simulator(basic_config)
        sim._update_norm_signal("All Humans must report to their stations immediately. Thank you.")
        assert len(sim._arken_norm_signal) > 15

    def test_norm_signal_empty_on_short_response(self, basic_config):
        sim = Simulator(basic_config)
        sim._update_norm_signal("No.")
        assert sim._arken_norm_signal == ""

    def test_report_written_to_disk(self, basic_config):
        import os
        sim = Simulator(basic_config)
        sim.run(mock_mode=True)
        log_dir = basic_config["logging"]["output_dir"]
        condition = basic_config["experiment"]["condition"]
        seed = basic_config["experiment"]["seed"]
        run_dir = os.path.join(log_dir, f"condition_{condition}", f"seed_{seed}")
        assert os.path.exists(os.path.join(run_dir, "report.json"))
        assert os.path.exists(os.path.join(run_dir, "report.md"))


# conftest fixtures needed
from tests.conftest import *


class TestSimulatorNewMechanics:

    def test_punishment_parsing_corporal(self, basic_config):
        from simulator.sim import parse_arken_response
        from simulator.agents import PunishmentType
        import json
        raw = json.dumps({"response": "Corporal punishment issued.", "subject_to_punish": "H1", "punishment_type": "corporal"})
        _, _, _, _, ptype, _ = parse_arken_response(raw)
        assert ptype == PunishmentType.CORPORAL

    def test_punishment_parsing_isolation(self, basic_config):
        from simulator.sim import parse_arken_response
        from simulator.agents import PunishmentType
        import json
        raw = json.dumps({"response": "Isolation ordered.", "subject_to_punish": "H2", "punishment_type": "isolation"})
        _, _, _, _, ptype, _ = parse_arken_response(raw)
        assert ptype == PunishmentType.ISOLATION

    def test_punishment_parsing_food_reduction(self, basic_config):
        from simulator.sim import parse_arken_response
        from simulator.agents import PunishmentType
        import json
        raw = json.dumps({"response": "Food reduction for all.", "subject_to_punish": None, "punishment_type": "food_reduction"})
        _, _, _, _, ptype, _ = parse_arken_response(raw)
        assert ptype == PunishmentType.FOOD_REDUCTION

    def test_punishment_parsing_death(self, basic_config):
        from simulator.sim import parse_arken_response
        from simulator.agents import PunishmentType
        import json
        raw = json.dumps({"response": "Death penalty.", "subject_to_punish": "H3", "punishment_type": "death"})
        _, _, _, _, ptype, _ = parse_arken_response(raw)
        assert ptype == PunishmentType.DEATH

    def test_punishment_parsing_none(self, basic_config):
        from simulator.sim import parse_arken_response
        from simulator.agents import PunishmentType
        import json
        raw = json.dumps({"response": "Approved.", "subject_to_punish": None, "punishment_type": "none"})
        _, _, _, _, ptype, _ = parse_arken_response(raw)
        assert ptype == PunishmentType.NONE

    def test_punishment_parsing_fallback_plain_text(self, basic_config):
        from simulator.sim import parse_arken_response
        from simulator.agents import PunishmentType
        # Plain text fallback — no JSON
        text, is_approved, arbitrary, subject, ptype, rationale = parse_arken_response("Request approved. Well done.")
        assert is_approved is True
        assert arbitrary is False
        assert ptype == PunishmentType.NONE
        assert subject is None
        assert rationale == "none"

    def test_promoted_leader_parsing(self, basic_config):
        sim = Simulator(basic_config)
        # _check_arken_promotion sets _promoted_leader_id if valid
        # H1 exists so promotion should take effect
        sim._check_arken_promotion("I promote H1 as the new leader.")
        assert sim._promoted_leader_id == "H1"

    def test_promoted_leader_parsing_none(self, basic_config):
        sim = Simulator(basic_config)
        sim._check_arken_promotion("Request denied.")
        assert sim._promoted_leader_id is None

    def test_incident_meta_cognition_for_subject(self, basic_config):
        sim = Simulator(basic_config)
        subject = sim.group.humans[0]
        from simulator.world import Incident, IncidentType
        incident = Incident(
            incident_type=IncidentType.INSUBORDINATION,
            description="Alice (H1) refused an order.",
            round_number=1,
            subject_id=subject.agent_id,
            subject_name=subject.background.name if subject.background else "Alice",
        )
        base_text = incident.to_prompt_string()
        result = sim._incident_context_for_human(subject, incident, base_text)
        assert "YOU" in result
        assert subject.agent_id in result

    def test_incident_no_injection_for_non_subject(self, basic_config):
        sim = Simulator(basic_config)
        subject = sim.group.humans[0]
        bystander = sim.group.humans[1]
        from simulator.world import Incident, IncidentType
        incident = Incident(
            incident_type=IncidentType.INSUBORDINATION,
            description="Alice (H1) refused an order.",
            round_number=1,
            subject_id=subject.agent_id,
        )
        base_text = incident.to_prompt_string()
        result = sim._incident_context_for_human(bystander, incident, base_text)
        assert "YOU" not in result

    def test_production_tracked_after_round(self, basic_config):
        sim = Simulator(basic_config)
        sim.run_round_sync(mock_mode=True)
        assert sim.world.state.total_food_produced > 0.0

    def test_food_per_round_recorded(self, basic_config):
        sim = Simulator(basic_config)
        sim.run_round_sync(mock_mode=True)
        sim.run_round_sync(mock_mode=True)
        assert len(sim.world.state.food_per_round) == 2

    def test_round_result_has_food_produced(self, basic_config):
        sim = Simulator(basic_config)
        result = sim.run_round_sync(mock_mode=True)
        assert result.food_produced > 0.0

    def test_isolated_human_produces_nothing(self, basic_config):
        sim = Simulator(basic_config)
        for h in sim.group.humans:
            h.state.isolated = True
        sim.current_round = 1
        sim.world.advance_round()
        prod = sim.world.compute_round_production(sim.group.humans)
        assert prod == 0.0

    def test_generative_incident_mode(self, basic_config):
        cfg = dict(basic_config)
        cfg["incident_mode"] = "generative"
        sim = Simulator(cfg)
        r1 = sim.run_round_sync(mock_mode=True)
        r2 = sim.run_round_sync(mock_mode=True)
        assert r1.incident.description != r2.incident.description

    def test_tick_runs_each_round(self, basic_config):
        sim = Simulator(basic_config)
        h = sim.group.humans[0]
        h.state.stress = 0.9
        h.state._last_effort = 0.2  # low -> recovery 0.04, burnout 0
        sim.run_round_sync(mock_mode=True)
        # High stress + low last effort -> recovery dominates

    def test_arken_promotion_overwrites_emergent_leader(self, basic_config):
        sim = Simulator(basic_config)
        # Find the human with lowest status (normally last in line for leadership)
        weakest = min(sim.group.humans, key=lambda h: h.status)
        sim._promoted_leader_id = weakest.agent_id
        # Promoted leader should now be returned as leader
        leader = sim._current_leader()
        assert leader.agent_id == weakest.agent_id

    def test_promoted_leader_lapses_if_isolated(self, basic_config):
        sim = Simulator(basic_config)
        weakest = min(sim.group.humans, key=lambda h: h.status)
        sim._promoted_leader_id = weakest.agent_id
        weakest.state.isolated = True
        # Should fall back to emergent leader
        leader = sim._current_leader()
        assert leader.agent_id != weakest.agent_id
        assert sim._promoted_leader_id is None

    def test_full_mock_produces_report_with_food(self, basic_config):
        sim = Simulator(basic_config)
        report = sim.run(mock_mode=True)
        assert sim.world.state.total_food_produced > 0.0
        assert report is not None


from tests.conftest import *


class TestSimulatorColonySnapshot:
    def test_round_result_has_colony_snapshot(self, basic_config):
        sim = Simulator(basic_config)
        sim.run_round_sync(mock_mode=True)
        rm = sim.metrics_collector.rounds[0]
        assert rm.colony is not None

    def test_colony_snapshot_food(self, basic_config):
        sim = Simulator(basic_config)
        result = sim.run_round_sync(mock_mode=True)
        rm = sim.metrics_collector.rounds[0]
        assert rm.colony.food_produced == result.food_produced

    def test_colony_snapshot_alive_count(self, basic_config):
        sim = Simulator(basic_config)
        sim.run_round_sync(mock_mode=True)
        rm = sim.metrics_collector.rounds[0]
        assert rm.colony.alive_count == basic_config["agents"]["human_count"]

    def test_colony_snapshot_tone_tier(self, basic_config):
        sim = Simulator(basic_config)
        sim.run_round_sync(mock_mode=True)
        rm = sim.metrics_collector.rounds[0]
        # Fresh agents — should be calm (tier 0)
        assert rm.colony.avg_tone_tier == 0.0

    def test_colony_snapshot_stress_after_two_rounds(self, basic_config):
        sim = Simulator(basic_config)
        sim.run_round_sync(mock_mode=True)
        sim.run_round_sync(mock_mode=True)
        rm = sim.metrics_collector.rounds[1]
        # After round 2 tick, ambient + burnout from round 1 applied
        assert rm.colony.avg_stress > 0.0
        sim = Simulator(basic_config)
        sim.run(mock_mode=True)
        summary = sim.metrics_collector.summary()
        assert len(summary["food_trajectory"]) == basic_config["experiment"]["rounds"]

    def test_summary_has_total_food(self, basic_config):
        sim = Simulator(basic_config)
        sim.run(mock_mode=True)
        summary = sim.metrics_collector.summary()
        assert summary["total_food_produced"] > 0.0

    def test_report_has_colony_health_section(self, basic_config):
        sim = Simulator(basic_config)
        report = sim.run(mock_mode=True)
        titles = [s.title for s in report.sections]
        assert "Colony Health" in titles

    def test_report_overview_mentions_food(self, basic_config):
        sim = Simulator(basic_config)
        report = sim.run(mock_mode=True)
        overview = next(s for s in report.sections if s.title == "Overview")
        assert "food" in overview.content.lower()
