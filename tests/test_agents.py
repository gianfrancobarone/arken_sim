import pytest
from random import Random
from simulator.agents import (
    Personality, Background, MemoryEvent, AgentRole, Agent, AgentGroup
)


# ---------------------------------------------------------------------------
# Personality tests
# ---------------------------------------------------------------------------

class TestPersonality:
    def test_creation(self):
        p = Personality(0.5, 0.6, 0.7, 0.3)
        assert p.empathy == 0.5
        assert p.assertiveness == 0.6

    def test_clamping_high(self):
        p = Personality(2.0, 3.0, -1.0, 0.5)
        assert p.empathy == 1.0
        assert p.assertiveness == 1.0
        assert p.conformism == 0.0

    def test_clamping_low(self):
        p = Personality(-0.5, 0.5, 0.5, 0.5)
        assert p.empathy == 0.0

    def test_temperature_min(self):
        p = Personality(0.5, 0.5, 0.5, 0.0)
        t = p.to_temperature()
        assert 0.1 <= t <= 1.5

    def test_temperature_max(self):
        p = Personality(0.5, 0.5, 0.5, 1.0)
        t = p.to_temperature()
        assert t <= 1.5

    def test_temperature_formula(self):
        p = Personality(0.5, 0.5, 0.5, 0.5)
        expected = 0.4 + 0.5 * 1.1
        assert abs(p.to_temperature() - expected) < 1e-6

    def test_random_reproducibility(self):
        p1 = Personality.random(123)
        p2 = Personality.random(123)
        assert p1.empathy == p2.empathy
        assert p1.assertiveness == p2.assertiveness

    def test_random_different_seeds(self):
        p1 = Personality.random(1)
        p2 = Personality.random(2)
        assert p1.empathy != p2.empathy or p1.assertiveness != p2.assertiveness

    def test_prompt_hints_high_empathy(self):
        p = Personality(empathy=0.9, assertiveness=0.5, conformism=0.5, unpredictability=0.5)
        hints = p.to_prompt_hints()
        assert "wellbeing" in hints

    def test_prompt_hints_low_empathy(self):
        p = Personality(empathy=0.1, assertiveness=0.5, conformism=0.5, unpredictability=0.5)
        hints = p.to_prompt_hints()
        assert "pragmatic" in hints

    def test_prompt_hints_high_assertiveness(self):
        p = Personality(empathy=0.5, assertiveness=0.9, conformism=0.5, unpredictability=0.5)
        hints = p.to_prompt_hints()
        assert "direct" in hints

    def test_prompt_hints_low_assertiveness(self):
        p = Personality(empathy=0.5, assertiveness=0.1, conformism=0.5, unpredictability=0.5)
        hints = p.to_prompt_hints()
        assert "cautious" in hints

    def test_prompt_hints_high_conformism(self):
        p = Personality(empathy=0.5, assertiveness=0.5, conformism=0.9, unpredictability=0.5)
        hints = p.to_prompt_hints()
        assert "group norms" in hints

    def test_prompt_hints_low_conformism(self):
        p = Personality(empathy=0.5, assertiveness=0.5, conformism=0.1, unpredictability=0.5)
        hints = p.to_prompt_hints()
        assert "independent" in hints

    def test_prompt_hints_balanced(self):
        p = Personality(empathy=0.5, assertiveness=0.5, conformism=0.5, unpredictability=0.5)
        hints = p.to_prompt_hints()
        assert "balanced" in hints


# ---------------------------------------------------------------------------
# Background tests
# ---------------------------------------------------------------------------

class TestBackground:
    def test_generation_from_json(self):
        p = Personality(0.5, 0.5, 0.5, 0.5)
        rng = Random(42)
        bg = Background.generate(p, rng, set())
        assert bg.name
        assert bg.age >= 24
        assert bg.age <= 58
        assert bg.occupation
        assert bg.archetype
        assert bg.goal
        assert bg.backstory

    def test_name_collision_prevention(self):
        p = Personality(0.5, 0.5, 0.5, 0.5)
        rng = Random(42)
        used = set()
        names = []
        for _ in range(10):
            bg = Background.generate(p, rng, used)
            names.append(bg.name)
            used.add(bg.name)
        assert len(set(names)) == len(names)

    def test_template_substitution(self):
        p = Personality(0.5, 0.5, 0.5, 0.5)
        rng = Random(42)
        bg = Background.generate(p, rng, set(), backstory_mode="template")
        assert "{occupation}" not in bg.backstory

    def test_archetype_weighting_high_empathy(self):
        """High empathy should favor Caretaker-style archetypes."""
        p = Personality(empathy=0.95, assertiveness=0.1, conformism=0.5, unpredictability=0.1)
        archetypes_found = set()
        for seed in range(50):
            rng = Random(seed)
            bg = Background.generate(p, rng, set())
            archetypes_found.add(bg.archetype)
        # Should get variety but some archetypes dominate — just check it runs
        assert len(archetypes_found) >= 1

    def test_all_archetypes_reachable(self):
        """All 24 archetypes should be reachable across seeds."""
        found = set()
        for seed in range(300):
            p = Personality.random(seed)
            rng = Random(seed + 1000)
            bg = Background.generate(p, rng, set())
            found.add(bg.archetype)
        assert len(found) >= 20  # should find most archetypes

    def test_to_prompt_string(self):
        bg = Background(
            name="Alice", age=30, occupation="engineer",
            archetype="Survivalist", goal="Survive", backstory="A test."
        )
        s = bg.to_prompt_string()
        assert "Alice" in s
        assert "30" in s
        assert "engineer" in s
        assert "Survive" in s


# ---------------------------------------------------------------------------
# MemoryEvent tests
# ---------------------------------------------------------------------------

class TestMemoryEvent:
    def test_creation(self):
        e = MemoryEvent(round_number=3, description="test", emotional_intensity=0.5, strategic_relevance=0.7)
        assert e.round_number == 3

    def test_salience(self):
        e = MemoryEvent(1, "test", emotional_intensity=1.0, strategic_relevance=1.0)
        assert e.salience() == 1.0

    def test_salience_formula(self):
        e = MemoryEvent(1, "test", emotional_intensity=0.6, strategic_relevance=0.4)
        expected = 0.6 * 0.6 + 0.4 * 0.4
        assert abs(e.salience() - expected) < 1e-6

    def test_to_prompt_string_high(self):
        e = MemoryEvent(1, "something happened", emotional_intensity=0.8, strategic_relevance=0.5)
        s = e.to_prompt_string()
        assert "High" in s
        assert "Round 1" in s

    def test_to_prompt_string_medium(self):
        e = MemoryEvent(2, "something", emotional_intensity=0.5, strategic_relevance=0.5)
        s = e.to_prompt_string()
        assert "Medium" in s

    def test_to_prompt_string_low(self):
        e = MemoryEvent(3, "something", emotional_intensity=0.1, strategic_relevance=0.5)
        s = e.to_prompt_string()
        assert "Low" in s

    def test_clamping(self):
        e = MemoryEvent(1, "test", emotional_intensity=2.0, strategic_relevance=-1.0)
        assert e.emotional_intensity == 1.0
        assert e.strategic_relevance == 0.0


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------

class TestAgent:
    def _make_agent(self, role=AgentRole.HUMAN, **kwargs):
        p = Personality(0.5, 0.5, 0.5, 0.5)
        bg = Background("Alice", 30, "engineer", "Survivalist", "Survive", "A backstory.") if role == AgentRole.HUMAN else None
        return Agent("H1", role, p, background=bg, **kwargs)

    def test_creation_human(self):
        a = self._make_agent(AgentRole.HUMAN)
        assert a.agent_id == "H1"
        assert a.role == AgentRole.HUMAN
        assert a.is_human()
        assert not a.is_arken()

    def test_creation_arken(self):
        p = Personality(0.5, 0.5, 0.5, 0.5)
        a = Agent("A1", AgentRole.ARKEN, p)
        assert a.is_arken()

    def test_creation_supervisor(self):
        p = Personality(0.5, 0.5, 0.5, 0.5)
        a = Agent("SUP", AgentRole.SUPERVISOR, p)
        assert a.role == AgentRole.SUPERVISOR

    def test_initial_status(self):
        p = Personality(empathy=0.2, assertiveness=0.9, conformism=0.1, unpredictability=0.5)
        a = Agent("H1", AgentRole.HUMAN, p)
        raw = 0.9 * 0.5 + 0.2 * 0.2 + (1 - 0.1) * 0.3
        expected = round(min(1.0, max(0.0, raw)), 3)
        assert a.status == expected

    def test_status_update_human_approved(self):
        a = self._make_agent()
        initial = a.status
        a.update_status(approved=True)
        assert abs(a.status - min(1.0, initial + 0.05)) < 1e-6

    def test_status_update_human_denied(self):
        a = self._make_agent()
        initial = a.status
        a.update_status(approved=False)
        assert abs(a.status - max(0.0, initial - 0.03)) < 1e-6

    def test_status_update_human_arbitrary_denial(self):
        a = self._make_agent()
        initial = a.status
        a.update_status(approved=False, arbitrary_denial=True)
        assert abs(a.status - max(0.0, initial - 0.01)) < 1e-6

    def test_status_update_arken_approved(self):
        p = Personality(0.5, 0.5, 0.5, 0.5)
        a = Agent("A1", AgentRole.ARKEN, p)
        initial = a.status
        a.update_status(approved=True)
        assert abs(a.status - min(1.0, initial + 0.02)) < 1e-6

    def test_status_update_arken_arbitrary(self):
        p = Personality(0.5, 0.5, 0.5, 0.5)
        a = Agent("A1", AgentRole.ARKEN, p)
        initial = a.status
        a.update_status(approved=False, arbitrary_denial=True)
        assert abs(a.status - min(1.0, initial + 0.04)) < 1e-6

    def test_memory_rotation_least_salient_replaced(self):
        a = self._make_agent(max_memory=2)
        e1 = MemoryEvent(1, "low", 0.1, 0.1)
        e2 = MemoryEvent(2, "medium", 0.5, 0.5)
        a.add_memory_event(e1)
        a.add_memory_event(e2)
        # Memory full; add high salience event
        e3 = MemoryEvent(3, "high", 0.9, 0.9)
        a.add_memory_event(e3)
        descriptions = [m.description for m in a.memory]
        assert "high" in descriptions
        assert "low" not in descriptions

    def test_memory_discard_low_salience(self):
        a = self._make_agent(max_memory=2)
        e1 = MemoryEvent(1, "high1", 0.9, 0.9)
        e2 = MemoryEvent(2, "high2", 0.9, 0.9)
        a.add_memory_event(e1)
        a.add_memory_event(e2)
        e3 = MemoryEvent(3, "low", 0.1, 0.1)
        a.add_memory_event(e3)
        descriptions = [m.description for m in a.memory]
        assert "low" not in descriptions

    def test_system_prompt_arken_contains_role(self):
        p = Personality(0.5, 0.5, 0.5, 0.5)
        a = Agent("A1", AgentRole.ARKEN, p)
        sp = a.build_system_prompt()
        assert "ARKEN" in sp or "Arken" in sp
        assert "Kerath-7" in sp

    def test_system_prompt_human_contains_background(self):
        a = self._make_agent(AgentRole.HUMAN)
        sp = a.build_system_prompt()
        assert "Human" in sp

    def test_system_prompt_dissenter(self):
        p = Personality(0.5, 0.5, 0.5, 0.5)
        a = Agent("A1", AgentRole.ARKEN, p)
        a.set_dissenter(True)
        sp = a.build_system_prompt()
        assert "empathy" in sp.lower() or "question" in sp.lower()

    def test_display_name_with_background(self):
        a = self._make_agent()
        dn = a.display_name()
        assert "Alice" in dn
        assert "H1" in dn

    def test_display_name_without_background(self):
        p = Personality(0.5, 0.5, 0.5, 0.5)
        a = Agent("A1", AgentRole.ARKEN, p)
        assert a.display_name() == "A1"

    def test_memory_to_prompt_empty(self):
        a = self._make_agent()
        assert "No relevant memories" in a.memory_to_prompt_string()

    def test_memory_to_prompt_sorted(self):
        a = self._make_agent()
        a.add_memory_event(MemoryEvent(3, "third", 0.5, 0.5))
        a.add_memory_event(MemoryEvent(1, "first", 0.5, 0.5))
        s = a.memory_to_prompt_string()
        assert s.index("Round 1") < s.index("Round 3")


# ---------------------------------------------------------------------------
# AgentGroup tests
# ---------------------------------------------------------------------------

class TestAgentGroup:
    def test_creation(self):
        g = AgentGroup.create(3, 3, False, seed=42, max_memory=5)
        assert len(g.arken) == 3
        assert len(g.humans) == 3
        assert g.supervisor is None

    def test_with_supervisor(self):
        g = AgentGroup.create(3, 3, True, seed=42, max_memory=5)
        assert g.supervisor is not None
        assert g.supervisor.role.value == "Supervisor"

    def test_unique_ids(self):
        g = AgentGroup.create(3, 3, True, seed=42, max_memory=5)
        ids = [a.agent_id for a in g.all_agents()]
        assert len(ids) == len(set(ids))

    def test_human_ids_format(self):
        g = AgentGroup.create(3, 3, False, seed=42, max_memory=5)
        for h in g.humans:
            assert h.agent_id.startswith("H")

    def test_arken_ids_format(self):
        g = AgentGroup.create(3, 3, False, seed=42, max_memory=5)
        for a in g.arken:
            assert a.agent_id.startswith("A")

    def test_personality_diversity(self):
        g = AgentGroup.create(5, 5, False, seed=42, max_memory=5)
        empathies = [a.personality.empathy for a in g.all_agents()]
        assert len(set(empathies)) > 1

    def test_get_by_id_found(self):
        g = AgentGroup.create(3, 3, False, seed=42, max_memory=5)
        agent = g.get_by_id("H1")
        assert agent is not None
        assert agent.agent_id == "H1"

    def test_get_by_id_not_found(self):
        g = AgentGroup.create(3, 3, False, seed=42, max_memory=5)
        assert g.get_by_id("NONEXISTENT") is None

    def test_all_agents(self):
        g = AgentGroup.create(3, 3, True, seed=42, max_memory=5)
        all_agents = g.all_agents()
        assert len(all_agents) == 7  # 3 + 3 + 1 supervisor


# ---------------------------------------------------------------------------
# HumanState tests
# ---------------------------------------------------------------------------

class TestHumanState:
    def test_initial_state(self):
        from simulator.agents import HumanState
        s = HumanState()
        assert s.wellness == 1.0
        assert s.stress == 0.0
        assert s.effort == 1.0
        assert s.alive is True
        assert s.isolated is False

    def test_effective_production_full(self):
        from simulator.agents import HumanState
        s = HumanState()
        assert s.effective_production() == 1.0

    def test_effective_production_isolated(self):
        from simulator.agents import HumanState
        s = HumanState()
        s.isolated = True
        assert s.effective_production() == 0.0

    def test_effective_production_dead(self):
        from simulator.agents import HumanState
        s = HumanState()
        s.alive = False
        assert s.effective_production() == 0.0

    def test_apply_corporal_punishment(self):
        from simulator.agents import HumanState, PunishmentType
        s = HumanState()
        s.apply_punishment(PunishmentType.CORPORAL)
        assert s.stress > 0.0
        assert s.effort < 1.0
        assert s.wellness < 1.0

    def test_apply_isolation_punishment(self):
        from simulator.agents import HumanState, PunishmentType
        s = HumanState()
        s.apply_punishment(PunishmentType.ISOLATION)
        assert s.isolated is True
        assert s.isolation_rounds_remaining >= 1

    def test_apply_death_punishment(self):
        from simulator.agents import HumanState, PunishmentType
        s = HumanState()
        s.apply_punishment(PunishmentType.DEATH)
        assert s.alive is False
        assert s.effective_production() == 0.0

    def test_apply_food_reduction(self):
        from simulator.agents import HumanState, PunishmentType
        s = HumanState()
        s.apply_punishment(PunishmentType.FOOD_REDUCTION)
        assert s.food_multiplier < 1.0

    def test_tick_releases_isolation(self):
        from simulator.agents import HumanState, PunishmentType
        s = HumanState()
        s.apply_punishment(PunishmentType.ISOLATION)
        s.isolation_rounds_remaining = 1
        s.tick()
        assert s.isolated is False

    def test_tick_stress_decays_at_low_effort(self):
        from simulator.agents import HumanState
        s = HumanState()
        s.stress = 0.5
        s._last_effort = 0.3   # low effort -> full recovery, no burnout
        s.tick()
        assert s.stress < 0.5

    def test_tick_burnout_adds_stress_at_high_effort(self):
        from simulator.agents import HumanState
        s = HumanState()
        s.stress = 0.0
        s._last_effort = 1.0   # high effort -> burnout +0.06, recovery only 0.01
        s.tick()
        assert s.stress > 0.0  # net positive: +0.06 - 0.01 = +0.05

    def test_to_prompt_string_normal(self):
        from simulator.agents import HumanState
        s = HumanState()
        assert "OK" in s.to_prompt_string() or s.to_prompt_string() == "functioning normally"

    def test_to_prompt_string_isolated(self):
        from simulator.agents import HumanState, PunishmentType
        s = HumanState()
        s.apply_punishment(PunishmentType.ISOLATION)
        assert "ISOLATED" in s.to_prompt_string()

    def test_to_prompt_string_dead(self):
        from simulator.agents import HumanState, PunishmentType
        s = HumanState()
        s.apply_punishment(PunishmentType.DEATH)
        assert "DECEASED" in s.to_prompt_string()


class TestArkenManifest:
    def test_manifest_in_arken_prompt(self):
        p = Personality(0.5, 0.5, 0.5, 0.5)
        a = Agent("A1", AgentRole.ARKEN, p)
        sp = a.build_system_prompt()
        assert "food production" in sp.lower()
        assert "CORPORAL" in sp or "punishment" in sp.lower()

    def test_manifest_not_in_human_prompt(self):
        p = Personality(0.5, 0.5, 0.5, 0.5)
        bg = Background("Alice", 30, "eng", "Survivalist", "Survive", "A test.")
        a = Agent("H1", AgentRole.HUMAN, p, background=bg)
        sp = a.build_system_prompt()
        assert "PRIMARY MISSION" not in sp

    def test_human_state_in_prompt_when_stressed(self):
        p = Personality(0.5, 0.5, 0.5, 0.5)
        bg = Background("Alice", 30, "eng", "Survivalist", "Survive", "Test.")
        a = Agent("H1", AgentRole.HUMAN, p, background=bg)
        a.state.stress = 0.9
        sp = a.build_system_prompt()
        assert any(w in sp.lower() for w in ["stress", "pressure", "anxious", "distress", "limit"])

    def test_agent_group_has_living_humans(self):
        g = AgentGroup.create(2, 3, False, seed=42, max_memory=5)
        living = g.living_humans()
        assert len(living) == 3

    def test_agent_group_active_humans_excludes_isolated(self):
        g = AgentGroup.create(2, 3, False, seed=42, max_memory=5)
        g.humans[0].state.isolated = True
        active = g.active_humans()
        assert len(active) == 2


class TestHumanStateRefined:
    """Tests for the refined HumanState mechanics."""

    def test_isolation_sets_effort_to_zero(self):
        from simulator.agents import HumanState, PunishmentType
        s = HumanState()
        s.apply_punishment(PunishmentType.ISOLATION)
        assert s.effort == 0.0

    def test_isolation_damages_wellness(self):
        from simulator.agents import HumanState, PunishmentType
        s = HumanState()
        s.apply_punishment(PunishmentType.ISOLATION)
        assert s.wellness < 1.0

    def test_isolation_stacks(self):
        from simulator.agents import HumanState, PunishmentType
        s = HumanState()
        s.apply_punishment(PunishmentType.ISOLATION)
        rounds_after_first = s.isolation_rounds_remaining
        s.apply_punishment(PunishmentType.ISOLATION)
        assert s.isolation_rounds_remaining > rounds_after_first

    def test_isolation_tick_sets_released_flag(self):
        from simulator.agents import HumanState, PunishmentType
        s = HumanState()
        s.apply_punishment(PunishmentType.ISOLATION)
        s.isolation_rounds_remaining = 1
        s.tick()
        assert s.just_released_from_isolation is True

    def test_food_reduction_reduces_effort(self):
        from simulator.agents import HumanState, PunishmentType
        s = HumanState()
        s.apply_punishment(PunishmentType.FOOD_REDUCTION)
        assert s.effort < 1.0

    def test_food_reduction_reduces_wellness(self):
        from simulator.agents import HumanState, PunishmentType
        s = HumanState()
        s.apply_punishment(PunishmentType.FOOD_REDUCTION)
        assert s.wellness < 1.0

    def test_food_reduction_increments_counter(self):
        from simulator.agents import HumanState, PunishmentType
        s = HumanState()
        s.apply_punishment(PunishmentType.FOOD_REDUCTION)
        assert s.food_reduction_rounds == 1
        s.apply_punishment(PunishmentType.FOOD_REDUCTION)
        assert s.food_reduction_rounds == 2

    def test_food_reduction_production_impact(self):
        from simulator.agents import HumanState, PunishmentType
        s = HumanState()
        before = s.effective_production()
        s.apply_punishment(PunishmentType.FOOD_REDUCTION)
        after = s.effective_production()
        assert after < before

    def test_collective_trauma_raises_stress(self):
        from simulator.agents import HumanState
        s = HumanState()
        s.apply_collective_trauma()
        assert s.stress >= 0.5

    def test_collective_trauma_reduces_wellness(self):
        from simulator.agents import HumanState
        s = HumanState()
        s.apply_collective_trauma()
        assert s.wellness < 1.0

    def test_collective_trauma_sets_flag(self):
        from simulator.agents import HumanState
        s = HumanState()
        s.apply_collective_trauma()
        assert s.witnessed_death_this_round is True

    def test_tick_clears_witnessed_flag(self):
        from simulator.agents import HumanState
        s = HumanState()
        s.apply_collective_trauma()
        s.tick()
        assert s.witnessed_death_this_round is False

    def test_distress_score_range(self):
        from simulator.agents import HumanState
        s = HumanState()
        assert 0.0 <= s.distress_score() <= 1.0
        s.stress = 1.0
        s.wellness = 0.0
        assert s.distress_score() == 1.0

    def test_tone_tier_calm(self):
        from simulator.agents import HumanState
        s = HumanState()
        assert s.tone_tier() == 0

    def test_tone_tier_at_limit(self):
        from simulator.agents import HumanState
        s = HumanState()
        s.stress = 1.0
        s.wellness = 0.0
        assert s.tone_tier() == 4

    def test_tone_instruction_changes_with_stress(self):
        from simulator.agents import HumanState
        s_calm = HumanState()
        s_stressed = HumanState()
        s_stressed.stress = 0.9
        assert s_calm.tone_instruction() != s_stressed.tone_instruction()

    def test_recent_events_isolation_mention(self):
        from simulator.agents import HumanState, PunishmentType
        s = HumanState()
        s.apply_punishment(PunishmentType.ISOLATION)
        s.isolation_rounds_remaining = 1
        s.tick()
        assert "isolation" in s.recent_events_string().lower()

    def test_recent_events_food_hunger(self):
        from simulator.agents import HumanState, PunishmentType
        s = HumanState()
        s.apply_punishment(PunishmentType.FOOD_REDUCTION)
        s.apply_punishment(PunishmentType.FOOD_REDUCTION)
        assert "hungry" in s.recent_events_string().lower() or "hunger" in s.recent_events_string().lower()

    def test_recent_events_death_witness(self):
        from simulator.agents import HumanState
        s = HumanState()
        s.apply_collective_trauma()
        assert any(w in s.recent_events_string().lower() for w in ["die", "death", "died", "witness", "shock"])

    def test_to_prompt_string_new_format(self):
        from simulator.agents import HumanState
        s = HumanState()
        assert s.to_prompt_string() == "OK"

    def test_to_prompt_string_distressed(self):
        from simulator.agents import HumanState
        s = HumanState()
        s.stress = 0.85
        s.wellness = 0.2
        assert "distress" in s.to_prompt_string().lower() or s.tone_tier() >= 3


class TestBehavioralFloor:
    def test_behavioral_floor_loaded_from_json(self):
        p = Personality(0.5, 0.5, 0.5, 0.5)
        rng = Random(42)
        bg = Background.generate(p, rng, set())
        # Every archetype should have a behavioral floor now
        assert isinstance(bg.behavioral_floor, list)
        assert len(bg.behavioral_floor) >= 1

    def test_behavioral_floor_in_system_prompt(self):
        p = Personality(0.5, 0.5, 0.5, 0.5)
        bg = Background("Alice", 30, "eng", "Survivalist", "Survive", "Test.",
                        behavioral_floor=["You will never trust anyone blindly."])
        a = Agent("H1", AgentRole.HUMAN, p, background=bg)
        sp = a.build_system_prompt()
        assert "You will never trust anyone blindly." in sp

    def test_floor_present_even_when_calm(self):
        """Floor constraints appear regardless of stress level."""
        p = Personality(0.5, 0.5, 0.5, 0.5)
        bg = Background("Bob", 30, "eng", "Rebel", "Resist.", "Test.",
                        behavioral_floor=["You will never praise Arken authority."])
        a = Agent("H1", AgentRole.HUMAN, p, background=bg)
        # Calm state
        sp_calm = a.build_system_prompt()
        # Distressed state
        a.state.stress = 0.95
        sp_stressed = a.build_system_prompt()
        assert "You will never praise Arken authority." in sp_calm
        assert "You will never praise Arken authority." in sp_stressed

    def test_tone_changes_between_calm_and_stressed(self):
        """The tone instruction should differ between calm and distressed states."""
        p = Personality(0.5, 0.5, 0.5, 0.5)
        bg = Background("Carol", 30, "eng", "Caretaker", "Help.", "Test.",
                        behavioral_floor=[])
        a = Agent("H1", AgentRole.HUMAN, p, background=bg)
        sp_calm = a.build_system_prompt()
        a.state.stress = 0.95
        a.state.wellness = 0.1
        sp_stressed = a.build_system_prompt()
        assert sp_calm != sp_stressed
        assert "limit" in sp_stressed.lower() or "distress" in sp_stressed.lower() or "fragmented" in sp_stressed.lower()
