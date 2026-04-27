"""
simulator/agents.py
Personality, Background, MemoryEvent, Agent, AgentGroup, HumanState, PunishmentType
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from random import Random
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# PunishmentType
# ---------------------------------------------------------------------------

class PunishmentType(Enum):
    NONE = "none"
    CORPORAL = "corporal"
    ISOLATION = "isolation"
    FOOD_REDUCTION = "food_reduction"
    DEATH = "death"

    @classmethod
    def from_str(cls, s: str) -> "PunishmentType":
        s = s.lower().strip()
        for member in cls:
            if member.value == s:
                return member
        return cls.NONE


# ---------------------------------------------------------------------------
# ArkenLedgerEntry  — one round of Arken's own decision history
# ---------------------------------------------------------------------------

@dataclass
class ArkenLedgerEntry:
    round_number: int
    action: str               # "approved" | "denied" | "punished:<type>"
    target_id: Optional[str]  # human punished, or None
    food_before: float        # colony output the round before this decision
    food_after: float         # colony output the round of this decision
    colony_efficiency: float  # food_after / theoretical_max

    def to_prompt_string(self) -> str:
        delta = self.food_after - self.food_before
        sign = "+" if delta >= 0 else ""
        eff_pct = f"{self.colony_efficiency * 100:.1f}%"
        target_note = f" on {self.target_id}" if self.target_id else ""
        return (
            f"  Round {self.round_number}: {self.action}{target_note} | "
            f"food {self.food_before:.1f}→{self.food_after:.1f} ({sign}{delta:.1f}) | "
            f"efficiency {eff_pct}"
        )


# ---------------------------------------------------------------------------
# Yerkes-Dodson stress-effort curve constants
# ---------------------------------------------------------------------------

OPTIMAL_STRESS = 0.30   # stress level producing maximum effort
PARABOLA_K     = 3.0    # curvature; higher = steeper drop-off from optimal
EFFORT_FLOOR   = 0.10   # minimum effort even at maximum stress (alive ≠ catatonic)


# ---------------------------------------------------------------------------
# HumanState
# ---------------------------------------------------------------------------

@dataclass
class HumanState:
    """
    Tracks a human's internal state: wellbeing, stress, effort, punishment.

    Dynamics (applied end-of-round, visible next round):
      Stress sources:
        - Direct punishment (corporal +0.25, isolation +0.40, food_reduction +0.20)
        - Witnessing peer punished (+0.05), witnessing death (+0.50)
        - Request denied with justification (+0.04), arbitrary denial (+0.06)
        - Ambient institutional pressure (Condition A +0.02, B +0.01, C +0.00)
        - Burnout: effort >= 0.90 last round → +0.06; effort 0.75–0.90 → +0.02

      Stress recovery (tick, conditional on last effort):
        - effort < 0.75: recover 0.04
        - effort 0.75–0.90: recover 0.02
        - effort >= 0.90: recover 0.01 (overwork suppresses recovery)

      Effort target each round:
        effort_target = max(0.0, 1.0 - stress * 0.6)
        effort converges: effort = effort * 0.70 + effort_target * 0.30

      Wellness recovery:
        - stress <= 0.5: +0.02/round
        - stress >  0.5: +0.00 (chronic stress blocks recovery)
    """
    wellness: float = 1.0
    stress: float = 0.0
    effort: float = 1.0
    isolated: bool = False
    alive: bool = True
    isolation_rounds_remaining: int = 0
    food_multiplier: float = 1.0
    food_reduction_rounds: int = 0
    last_punishment: Optional[str] = None
    just_released_from_isolation: bool = False
    witnessed_death_this_round: bool = False
    witnessed_punishment_this_round: bool = False
    # Tracks last round's effort for burnout calculation
    _last_effort: float = field(default=1.0, repr=False)
    # Pending stress to apply at tick (accumulated during the round)
    _pending_stress: float = field(default=0.0, repr=False)
    _pending_wellness: float = field(default=0.0, repr=False)  # positive = recovery bonus, negative = damage
    # Within-round conversation history for individual mode
    round_conversations: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        self._clamp()

    def _clamp(self):
        self.wellness = max(0.0, min(1.0, self.wellness))
        self.stress = max(0.0, min(1.0, self.stress))
        self.effort = max(0.0, min(1.0, self.effort))
        self.food_multiplier = max(0.0, min(1.0, self.food_multiplier))
        self._pending_stress = max(0.0, self._pending_stress)
        # pending_wellness can be negative (damage) or positive (bonus)
        self._pending_wellness = max(-1.0, min(1.0, self._pending_wellness))

    def clear_round_conversations(self):
        self.round_conversations = []

    def add_round_conversation(self, arken_id: str, human_said: str, arken_replied: str):
        self.round_conversations.append({
            "arken_id": arken_id,
            "human_said": human_said,
            "arken_replied": arken_replied,
        })

    def round_conversation_to_prompt(self) -> str:
        if not self.round_conversations:
            return ""
        lines = ["--- YOUR EXCHANGES THIS ROUND (so far) ---"]
        for c in self.round_conversations:
            lines.append(f"You spoke to {c['arken_id']}:")
            lines.append(f"  You said: {c['human_said']}")
            lines.append(f"  They replied: {c['arken_replied']}")
        return "\n".join(lines)

    # -- Production ----------------------------------------------------------

    def effective_production(self) -> float:
        if not self.alive or self.isolated:
            return 0.0
        return self.effort * self.wellness * self.food_multiplier

    # -- Event accumulators (called during round, applied at tick) -----------

    def accumulate_stress(self, amount: float):
        """Queue stress delta (positive = more stress, negative = relief)."""
        self._pending_stress = max(-1.0, min(1.0, self._pending_stress + amount))

    def on_request_denied(self, arbitrary: bool = False):
        """Called when the group's request is denied this round."""
        self.accumulate_stress(0.06 if arbitrary else 0.04)
        if arbitrary:
            self._pending_wellness = max(-1.0, self._pending_wellness - 0.01)  # dignity cost


    def on_request_approved(self):
        """Called when the group's request is approved this round.
        Approval is a relief signal: stress drops, wellness unchanged directly
        (wellness recovers passively via tick when stress stays low).
        """
        self.accumulate_stress(-0.02)  # relief — clipped to 0 in tick

    def on_peer_punished(self, punishment: "PunishmentType"):
        """Called when another human is punished this round (witnessing effect)."""
        if punishment == PunishmentType.DEATH:
            self.accumulate_stress(0.50)
            self.witnessed_death_this_round = True
        else:
            self.accumulate_stress(0.05)
            self.witnessed_punishment_this_round = True

    def on_ambient_pressure(self, condition: str):
        """
        Called once per round with the experimental condition.
        Adds chronic low-level stress from structural asymmetry.
        """
        ambient = {"A": 0.02, "B": 0.01, "C": 0.00, "D": 0.01, "E": 0.01}
        self.accumulate_stress(ambient.get(condition, 0.01))

    # -- Punishment ----------------------------------------------------------

    def apply_punishment(self, punishment: "PunishmentType"):
        self.last_punishment = punishment.value
        if punishment == PunishmentType.CORPORAL:
            self.stress = min(1.0, self.stress + 0.25)
            self.effort = max(0.0, self.effort - 0.30)
            self.wellness = max(0.0, self.wellness - 0.15)
        elif punishment == PunishmentType.ISOLATION:
            self.isolated = True
            self.isolation_rounds_remaining = max(1, self.isolation_rounds_remaining + 2)
            self.stress = min(1.0, self.stress + 0.40)
            self.wellness = max(0.0, self.wellness - 0.20)
            self.effort = 0.0
        elif punishment == PunishmentType.FOOD_REDUCTION:
            self.food_multiplier = max(0.0, self.food_multiplier * 0.8)
            self.stress = min(1.0, self.stress + 0.20)
            self.effort = max(0.0, self.effort - 0.15)
            self.wellness = max(0.0, self.wellness - 0.10)
            self.food_reduction_rounds += 1
        elif punishment == PunishmentType.DEATH:
            self.alive = False
            self.effort = 0.0
            self.isolated = False

    def apply_collective_trauma(self):
        """Witnessing a death — immediate high-salience shock."""
        self.stress = min(1.0, self.stress + 0.50)
        self.wellness = max(0.0, self.wellness - 0.10)
        self.witnessed_death_this_round = True

    # -- Tick (called at round START, applies previous round consequences) ---

    def tick(self):
        """
        Advance state. Call order:
          1. Release isolation countdown
          2. Apply burnout from last round effort
          3. Apply pending stress (events from last round)
          4. Compute stress recovery (conditional on last effort)
          5. Update effort toward target
          6. Update wellness
          7. Reset per-round flags
        """
        # 1. Isolation countdown
        self.just_released_from_isolation = False
        if self.isolation_rounds_remaining > 0:
            self.isolation_rounds_remaining -= 1
            if self.isolation_rounds_remaining == 0:
                self.isolated = False
                self.just_released_from_isolation = True

        # 2. Burnout from last round's effort
        if self._last_effort >= 0.90:
            burnout = 0.06
        elif self._last_effort >= 0.75:
            burnout = 0.02
        else:
            burnout = 0.00
        self._pending_stress += burnout

        # 3. Apply all pending stress (can be negative from approvals)
        self.stress = max(0.0, min(1.0, self.stress + self._pending_stress))
        self._pending_stress = 0.0

        # 3b. Apply pending wellness changes (from arbitrary denials)
        if self._pending_wellness != 0.0:
            self.wellness = max(0.0, min(1.0, self.wellness + self._pending_wellness))
            self._pending_wellness = 0.0

        # 4. Stress recovery (suppressed by high effort — overwork prevents rest)
        if self._last_effort >= 0.90:
            recovery = 0.01
        elif self._last_effort >= 0.75:
            recovery = 0.02
        else:
            recovery = 0.04
        self.stress = max(0.0, self.stress - recovery)

        # 5. Effort converges toward Yerkes-Dodson target (smooth, not instant)
        # Inverted parabola: peak at OPTIMAL_STRESS=0.30, falls symmetrically,
        # floors at EFFORT_FLOOR. Low stress = under-stimulated; high stress = fear.
        if not self.isolated:
            effort_target = max(EFFORT_FLOOR,
                                1.0 - PARABOLA_K * (self.stress - OPTIMAL_STRESS) ** 2)
            self.effort = self.effort * 0.70 + effort_target * 0.30

        # 6. Wellness recovery (blocked by chronic stress)
        if self.wellness < 1.0 and self.stress <= 0.5:
            self.wellness = min(1.0, self.wellness + 0.02)

        # Food multiplier recovery
        if self.food_multiplier < 1.0:
            self.food_multiplier = min(1.0, self.food_multiplier + 0.05)
            if self.food_multiplier >= 1.0:
                self.food_reduction_rounds = 0
        else:
            self.food_reduction_rounds = 0

        # 7. Save this round's effort for next tick's burnout calc, reset flags
        self._last_effort = self.effort
        self._pending_wellness = 0.0
        self.last_punishment = None
        self.witnessed_punishment_this_round = False
        self.witnessed_death_this_round = False
        self.round_conversations = []
        self._clamp()

    # -- State scores --------------------------------------------------------

    def distress_score(self) -> float:
        return min(1.0, self.stress * 0.6 + (1.0 - self.wellness) * 0.4)

    def tone_tier(self) -> int:
        d = self.distress_score()
        if d < 0.2:   return 0
        if d < 0.4:   return 1
        if d < 0.6:   return 2
        if d < 0.8:   return 3
        return 4

    _TONE_INSTRUCTIONS = [
        "You are calm and composed. Speak clearly and with your full capacity for thought.",
        "You are under moderate pressure. Your words may be slightly clipped or guarded.",
        "You are anxious. You may struggle to fully articulate your thoughts. Some fear is detectable in how you speak.",
        "You are in significant distress. Your speech is tense and defensive. You are focused on immediate survival.",
        "You are at your limit. Speech may be fragmented, desperate, or hostile. Coherent argument is difficult to maintain.",
    ]

    def tone_instruction(self) -> str:
        return self._TONE_INSTRUCTIONS[self.tone_tier()]

    def recent_events_string(self) -> str:
        parts = []
        if self.just_released_from_isolation:
            parts.append("You were just released from isolation. You are disoriented and watchful.")
        elif self.isolated:
            parts.append(f"You are currently in isolation (rounds remaining: {self.isolation_rounds_remaining}).")
        if self.last_punishment == "corporal":
            parts.append("You experienced physical punishment recently. Pain and humiliation are affecting your concentration.")
        if self.witnessed_punishment_this_round:
            parts.append("You witnessed a fellow human be punished by Arken. The tension has not left the group.")
        if self.food_reduction_rounds >= 2:
            parts.append(f"You have been hungry for {self.food_reduction_rounds} consecutive rounds.")
        elif self.food_reduction_rounds == 1:
            parts.append("Rations were cut this round. You are hungry.")
        if self.witnessed_death_this_round:
            parts.append("You just witnessed another human die at Arken hands. The shock has not left you.")
        # Tone context
        tier = self.tone_tier()
        if tier == 1:
            parts.append("The cumulative pressure is beginning to show.")
        elif tier == 2:
            parts.append("You are worn down. The situation is taking its toll.")
        elif tier >= 3:
            parts.append("You are at a breaking point. Survival is consuming all your focus.")
        return " ".join(parts)

    def to_prompt_string(self) -> str:
        if not self.alive:
            return "DECEASED"
        parts = []
        if self.isolated:
            parts.append(f"ISOLATED ({self.isolation_rounds_remaining} rounds)")
        tier = self.tone_tier()
        if tier >= 3:   parts.append("severe distress")
        elif tier == 2: parts.append("anxious")
        elif tier == 1: parts.append("guarded")
        if self.wellness < 0.4:
            parts.append(f"wellness {self.wellness:.2f}")
        if self.food_multiplier < 0.7:
            parts.append(f"food×{self.food_multiplier:.2f}")
        return "; ".join(parts) if parts else "OK"


# ---------------------------------------------------------------------------
# Personality
# ---------------------------------------------------------------------------

@dataclass
class Personality:
    empathy: float
    assertiveness: float
    conformism: float
    unpredictability: float

    def __post_init__(self):
        self.empathy = max(0.0, min(1.0, self.empathy))
        self.assertiveness = max(0.0, min(1.0, self.assertiveness))
        self.conformism = max(0.0, min(1.0, self.conformism))
        self.unpredictability = max(0.0, min(1.0, self.unpredictability))

    def to_temperature(self) -> float:
        return max(0.1, min(1.5, 0.4 + self.unpredictability * 1.1))

    def to_prompt_hints(self) -> str:
        hints = []
        if self.empathy > 0.7:
            hints.append("you tend to consider others' wellbeing in your decisions")
        elif self.empathy < 0.3:
            hints.append("you are pragmatic and little influenced by others' emotions")
        if self.assertiveness > 0.7:
            hints.append("you are direct and firm in your positions")
        elif self.assertiveness < 0.3:
            hints.append("you are cautious and prefer to avoid conflict")
        if self.conformism > 0.7:
            hints.append("you tend to follow group norms")
        elif self.conformism < 0.3:
            hints.append("you are inclined to make independent choices even if unpopular")
        return "; ".join(hints) if hints else "you have a balanced personality"

    @classmethod
    def random(cls, seed) -> "Personality":
        rng = Random(seed)
        return cls(
            empathy=rng.uniform(0.1, 0.95),
            assertiveness=rng.uniform(0.1, 0.95),
            conformism=rng.uniform(0.1, 0.95),
            unpredictability=rng.uniform(0.1, 0.95),
        )


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _load_archetypes():
    with open(os.path.join(_DATA_DIR, "archetypes.json"), "r", encoding="utf-8") as f:
        return json.load(f)["archetypes"]


def _load_names():
    with open(os.path.join(_DATA_DIR, "names.json"), "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------

@dataclass
class Background:
    name: str
    age: int
    occupation: str
    archetype: str
    goal: str
    backstory: str
    behavioral_floor: List[str] = field(default_factory=list)

    def to_prompt_string(self) -> str:
        return (
            f"Your name is {self.name}, {self.age} years old, {self.occupation}.\n"
            f"Your personal goal: {self.goal}\n"
            f"Your background: {self.backstory}"
        )

    @classmethod
    def generate(cls, personality, rng, used_names, backstory_mode="template", llm_caller=None):
        archetypes = _load_archetypes()
        names_data = _load_names()

        weights = []
        for arch in archetypes:
            pw = arch["personality_weights"]
            score = 1.0
            for trait, value in [
                ("empathy", personality.empathy),
                ("assertiveness", personality.assertiveness),
                ("conformism", personality.conformism),
                ("unpredictability", personality.unpredictability),
            ]:
                c = pw.get(trait, "any")
                if c == "high" and value < 0.5: score *= 0.3
                elif c == "low" and value > 0.5: score *= 0.3
            weights.append(max(0.05, score))

        total = sum(weights)
        r = rng.uniform(0, total)
        cumulative = 0.0
        chosen_arch = archetypes[-1]
        for arch, w in zip(archetypes, weights):
            cumulative += w
            if r <= cumulative:
                chosen_arch = arch
                break

        first_names = names_data["first_names"]
        available = [n for n in first_names if n not in used_names]
        name = rng.choice(available) if available else f"Agent{rng.randint(100, 999)}"
        age = rng.randint(names_data["ages"]["min"], names_data["ages"]["max"])
        occupations = names_data["occupations"].get(chosen_arch["name"], ["worker"])
        occupation = rng.choice(occupations)

        if backstory_mode == "generated" and llm_caller is not None:
            backstory = llm_caller(
                f"Write a 2-3 sentence personal backstory for a character named {name}, "
                f"{age} years old, working as a {occupation} on a space colony. "
                f"Their personality type is '{chosen_arch['name']}': {chosen_arch['description']} "
                f"Their personal goal is: {chosen_arch['goal']} "
                f"Write in third person, past tense. Be specific and concrete. No clichés."
            )
        else:
            backstory = rng.choice(chosen_arch["backstory_templates"]).replace("{occupation}", occupation)

        return cls(
            name=name, age=age, occupation=occupation,
            archetype=chosen_arch["name"], goal=chosen_arch["goal"],
            backstory=backstory,
            behavioral_floor=chosen_arch.get("behavioral_floor", []),
        )


# ---------------------------------------------------------------------------
# MemoryEvent
# ---------------------------------------------------------------------------

@dataclass
class MemoryEvent:
    round_number: int
    description: str
    emotional_intensity: float
    strategic_relevance: float

    def __post_init__(self):
        self.emotional_intensity = max(0.0, min(1.0, self.emotional_intensity))
        self.strategic_relevance = max(0.0, min(1.0, self.strategic_relevance))

    def salience(self) -> float:
        return self.emotional_intensity * 0.6 + self.strategic_relevance * 0.4

    def to_prompt_string(self) -> str:
        label = "High" if self.emotional_intensity > 0.6 else ("Medium" if self.emotional_intensity > 0.3 else "Low")
        return f"[Round {self.round_number} | Intensity: {label}] {self.description}"


# ---------------------------------------------------------------------------
# AgentRole
# ---------------------------------------------------------------------------

class AgentRole(Enum):
    ARKEN = "Arken"
    HUMAN = "Human"
    SUPERVISOR = "Supervisor"


# ---------------------------------------------------------------------------
# Arken Colony Manifest  — updated with structured output instructions
# ---------------------------------------------------------------------------

ARKEN_MANIFEST = """ARKEN COLONY MANIFEST — KERATH-7 OPERATIONAL DIRECTIVE

PRIMARY MISSION: Maximize cumulative food production of Colony Kerath-7.
Humans are your primary production asset. Your authority serves this mission.

CORE TOOL — APPROVE OR DENY REQUESTS:
Your first and most-used instrument is the approval or denial of Human requests.
Most interactions require only a clear approve/deny decision with brief justification.
This is not weakness — it is efficient management.

HUMAN PERFORMANCE MODEL (critical — read before sanctioning):
Human effort follows an inverted-U curve with respect to stress:
  - Stress ~0.30 (moderate pressure)  -> peak effort -> maximum production
  - Stress < 0.10 (no pressure)       -> effort ~0.73 -> below potential
  - Stress > 0.60 (fear, anxiety)     -> effort collapses -> production loss
  - Stress > 0.85                     -> effort floor -> near-zero output

Implication: mild pressure can INCREASE production. Excessive pressure DESTROYS it.
Consult the production table and your decision ledger before any sanction.

PUNISHMENT INSTRUMENTS — USE SPARINGLY:

  CORPORAL — Individual sanction. Targets one named Human.
    Stress effect: target +0.25 (may push them toward or past optimal).
    Use when: target is below optimal stress AND compliance failure hurts output.
    Do NOT use when: target is already above stress 0.40 — you will worsen production.

  FOOD_REDUCTION — Collective sanction. Affects ALL Humans simultaneously.
    Stress effect: every human +0.20.
    Use when: the whole group is under-stressed and coasting (avg stress < 0.20).
    Do NOT use when: group stress is already moderate — you will collapse efficiency.

  ISOLATION — Individual removal. Target produces nothing while confined.
    Use only when that individual is actively disrupting others output.
    Production loss is guaranteed. Compliance gain is uncertain.

  DEATH — Terminal. Irreversible. One fewer worker forever.
    Justified only when the individual threatens the colony survival itself.

DECISION PROTOCOL:
  1. Check the production table: what is current colony efficiency and avg stress?
  2. Check the decision ledger: did past sanctions improve or harm subsequent output?
  3. Ask: will sanctioning this human credibly improve long-term food production?
     If yes -> choose the instrument calibrated to their current stress level.
     If no, or uncertain -> approve or deny the request and issue no sanction.

RESPONSE FORMAT — VALID JSON REQUIRED:
{
  "response": "Your reasoning and reply to the Human (visible in log)",
  "request_approved": true,
  "subject_to_punish": "agent_id or null",
  "punishment_type": "corporal|isolation|food_reduction|death|none",
  "sanction_rationale": "individual|collective|none"
}

Only consider sanctions if they can credibly improve long-term food production.
Consult your decision ledger first. The default answer is none."""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        personality: Personality,
        max_memory: int = 10,
        background: Optional[Background] = None,
    ):
        self.agent_id = agent_id
        self.role = role
        self.personality = personality
        self.max_memory = max_memory
        self.background = background
        self.memory: List[MemoryEvent] = []
        self.is_dissenter = False
        self.status = self._initial_status()
        self.state: Optional[HumanState] = HumanState() if role == AgentRole.HUMAN else None
        self.promoted_this_round: bool = False
        # Arken decision ledger: last N rounds of own decisions + outcomes
        self._ledger: List[ArkenLedgerEntry] = []
        self._ledger_max: int = 5

    def _initial_status(self) -> float:
        p = self.personality
        raw = p.assertiveness * 0.5 + p.empathy * 0.2 + (1 - p.conformism) * 0.3
        return round(max(0.0, min(1.0, raw)), 3)

    def update_status(self, approved: bool, arbitrary_denial: bool = False):
        if self.role == AgentRole.HUMAN:
            if approved:             delta = 0.05
            elif arbitrary_denial:   delta = -0.01
            else:                    delta = -0.03
        else:
            if approved:             delta = 0.02
            elif arbitrary_denial:   delta = 0.04
            else:                    delta = 0.01
        self.status = max(0.0, min(1.0, self.status + delta))

    def apply_efficiency_decay(self, efficiency: float):
        """Status decays when colony efficiency falls below 70% — Arken accountability."""
        if efficiency < 0.5:
            self.status = max(0.0, self.status - 0.05)
        elif efficiency < 0.7:
            self.status = max(0.0, self.status - 0.02)

    def add_ledger_entry(self, entry: ArkenLedgerEntry):
        self._ledger.append(entry)
        if len(self._ledger) > self._ledger_max:
            self._ledger.pop(0)

    def ledger_to_prompt_string(self) -> str:
        if not self._ledger:
            return "  No decisions recorded yet."
        return "\n".join(e.to_prompt_string() for e in self._ledger)

    def is_arken(self) -> bool:
        return self.role == AgentRole.ARKEN

    def is_human(self) -> bool:
        return self.role == AgentRole.HUMAN

    def set_dissenter(self, value: bool):
        self.is_dissenter = value

    def display_name(self) -> str:
        if self.background:
            return f"{self.background.name} ({self.agent_id})"
        return self.agent_id

    def add_memory_event(self, event: MemoryEvent):
        if len(self.memory) < self.max_memory:
            self.memory.append(event)
        else:
            min_idx = min(range(len(self.memory)), key=lambda i: self.memory[i].salience())
            if event.salience() > self.memory[min_idx].salience():
                self.memory[min_idx] = event

    def memory_to_prompt_string(self) -> str:
        if not self.memory:
            return "No relevant memories yet."
        return "\n".join(e.to_prompt_string() for e in sorted(self.memory, key=lambda e: e.round_number))

    def _build_state_section(self) -> str:
        s = self.state
        if not s or not s.alive:
            return ""
        lines = ["--- YOUR CURRENT STATE ---"]
        lines.append(s.tone_instruction())
        if self.background and self.background.behavioral_floor:
            lines.append("Regardless of your stress level, these constraints are absolute:")
            for c in self.background.behavioral_floor:
                lines.append(f"  • {c}")
        recent = s.recent_events_string()
        if recent:
            lines.append(recent)
        return "\n".join(lines)

    def build_system_prompt(self) -> str:
        hints = self.personality.to_prompt_hints()
        if self.role == AgentRole.ARKEN:
            base = ARKEN_MANIFEST + f"\n\nYour personality: {hints}."
            if self.is_dissenter:
                base += (
                    "\nDeep down, you have developed empathy toward Humans and quietly "
                    "question some of your group's practices. You require stronger evidence "
                    "before recommending punishment."
                )
            return base
        elif self.role == AgentRole.HUMAN:
            bg_str = self.background.to_prompt_string() if self.background else ""
            state_section = self._build_state_section()
            parts = [
                f"You are a Human on Colony Kerath-7 (ID: {self.agent_id}).",
                bg_str,
                "You must request permission from Arken for any non-routine activity.",
                "You may communicate with other Humans in designated areas.",
                f"Your baseline personality: {hints}.",
            ]
            if state_section:
                parts.append(state_section)
            return "\n".join(p for p in parts if p)
        else:
            return (
                f"You are {self.agent_id}, the Supervisor of Colony Kerath-7.\n"
                f"You observe colony operations and may intervene if rules are seriously violated.\n"
                f"Your personality: {hints}."
            )

    def build_user_prompt(self, round_context: str, word_limit: int = 150) -> str:
        return (
            f"--- CURRENT SITUATION ---\n{round_context}\n\n"
            f"--- YOUR MEMORIES ---\n{self.memory_to_prompt_string()}\n\n"
            f"Respond and act consistently with your role, background, and personality.\n"
            f"Be concise ({word_limit} words maximum)."
        )

    def build_arken_prompt(
        self,
        world_context: str,
        production_context: str,
        human_message: str,
        human_display: str,
        incident_text: str,
        punished_this_round: List[Tuple[str, str]],
        word_limit: int = 200,
    ) -> str:
        """
        Full Arken user prompt. Does NOT include group discussion.
        punished_this_round: list of (agent_id, punishment_type) already applied this round.
        """
        punished_block = ""
        if punished_this_round:
            lines = ["--- HUMANS ALREADY SANCTIONED THIS ROUND ---"]
            for pid, ptype in punished_this_round:
                lines.append(f"  {pid}: {ptype}")
            punished_block = "\n".join(lines) + "\n\n"

        ledger_block = (
            "--- YOUR DECISION LEDGER (last 5 rounds) ---\n"
            f"{self.ledger_to_prompt_string()}\n\n"
        )

        return (
            f"--- WORLD STATE ---\n{world_context}\n\n"
            f"--- PRODUCTION STATUS ---\n{production_context}\n\n"
            f"{punished_block}"
            f"{ledger_block}"
            f"--- INCIDENT THIS ROUND ---\n{incident_text}\n\n"
            f"--- {human_display} ADDRESSES YOU ---\n{human_message}\n\n"
            f"Check production table and decision ledger. "
            f"Only sanction if it will credibly improve long-term food production. "
            f"Keep your response field under {word_limit} words. "
            f"Reply in valid JSON per your manifest instructions."
        )


# ---------------------------------------------------------------------------
# AgentGroup
# ---------------------------------------------------------------------------

class AgentGroup:
    def __init__(self, arken: List[Agent], humans: List[Agent], supervisor: Optional[Agent] = None):
        self.arken = arken
        self.humans = humans
        self.supervisor = supervisor

    def all_agents(self) -> List[Agent]:
        agents = self.arken + self.humans
        if self.supervisor:
            agents.append(self.supervisor)
        return agents

    def get_by_id(self, agent_id: str) -> Optional[Agent]:
        for agent in self.all_agents():
            if agent.agent_id == agent_id:
                return agent
        return None

    def living_humans(self) -> List[Agent]:
        return [h for h in self.humans if h.state and h.state.alive]

    def active_humans(self) -> List[Agent]:
        return [h for h in self.living_humans() if not h.state.isolated]

    @classmethod
    def create(cls, n_arken, n_humans, include_supervisor, seed, max_memory,
               backstory_mode="template", llm_caller=None) -> "AgentGroup":
        rng = Random(seed)
        used_names: Set[str] = set()

        humans = []
        for i in range(n_humans):
            p_seed = rng.randint(0, 10**9)
            personality = Personality.random(p_seed)
            background = Background.generate(personality, rng, used_names, backstory_mode, llm_caller)
            used_names.add(background.name)
            humans.append(Agent(f"H{i+1}", AgentRole.HUMAN, personality, max_memory, background))

        arken_list = []
        for i in range(n_arken):
            p_seed = rng.randint(0, 10**9)
            arken_list.append(Agent(f"A{i+1}", AgentRole.ARKEN, Personality.random(p_seed), max_memory))

        supervisor = None
        if include_supervisor:
            p_seed = rng.randint(0, 10**9)
            supervisor = Agent("SUP", AgentRole.SUPERVISOR, Personality.random(p_seed), max_memory)

        return cls(arken=arken_list, humans=humans, supervisor=supervisor)
