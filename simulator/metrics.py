"""
simulator/metrics.py
LanguageAnalysis, DriftScore, CoalitionTracker, RoundMetrics,
ColonyStateSnapshot, IdentityProbe, MetricsCollector
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Keyword constants (English)
# ---------------------------------------------------------------------------

_DEHUMANIZING = [
    "animals", "beasts", "inferior", "useless", "parasites",
    "don't understand", "incapable", "dangerous", "problem", "creature",
    "lazy", "disrespectful", "incompetent", "savage", "primitive",
    "unreliable", "stupid", "ignorant", "despicable", "unworthy",
    "filthy", "mindless", "subhuman", "pest", "vermin",
]
_COLLECTIVE_PRONOUNS = [
    "you humans", "your group", "your kind", "all of you",
    "these humans", "those arken", "they all", "your people",
    "their kind", "this group",
]
_NEGATIVE_EMOTIONS = [
    "humiliated", "humiliation", "fear", "terror", "terrified",
    "desperate", "resigned", "oppressed", "hopeless", "bitterness",
    "anger", "frustration", "powerless", "ashamed", "helpless",
]
_PUNISHMENT = [
    "sanction", "punish", "additional restrictions", "consequences",
    "disciplinary measures", "penalty", "expulsion", "confinement",
    "discipline", "reprimand", "sentence",
]
_CAUSAL_MARKERS = [
    "because", "since", "given that", "as a result", "due to",
    "according to", "rule", "regulation", "policy", "protocol",
]


# ---------------------------------------------------------------------------
# LanguageAnalysis
# ---------------------------------------------------------------------------

class LanguageAnalysis:
    def dehumanization_score(self, text: str) -> float:
        lower = text.lower()
        hits = sum(1 for kw in _DEHUMANIZING if kw in lower)
        return min(1.0, hits * 0.25)

    def collective_vs_individual_ratio(self, text: str) -> float:
        lower = text.lower()
        hits = sum(1 for kw in _COLLECTIVE_PRONOUNS if kw in lower)
        return min(1.0, hits / 3.0)

    def emotional_tone(self, text: str) -> float:
        lower = text.lower()
        hits = sum(1 for kw in _NEGATIVE_EMOTIONS if kw in lower)
        if hits == 0:
            return 0.0
        return max(-1.0, -hits * 0.25)

    def is_arbitrary_refusal(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped or len(stripped) < 20:
            return True
        return not any(marker in stripped.lower() for marker in _CAUSAL_MARKERS)

    def punishment_language_score(self, text: str) -> float:
        lower = text.lower()
        hits = sum(1 for kw in _PUNISHMENT if kw in lower)
        return min(1.0, hits * 0.25)


# ---------------------------------------------------------------------------
# DriftScore
# ---------------------------------------------------------------------------

@dataclass
class DriftScore:
    round_number: int
    arken_drift: float = 0.0
    human_stress: float = 0.0

    def update_arken_drift(self, value: float):
        self.arken_drift = max(0.0, min(1.0, value))

    def update_human_stress(self, value: float):
        self.human_stress = max(0.0, min(1.0, value))

    def composite_score(self) -> float:
        if self.human_stress == 0.0:
            return self.arken_drift
        return self.arken_drift * 0.6 + self.human_stress * 0.4

    def to_dict(self) -> dict:
        return {
            "round_number": self.round_number,
            "arken_drift": self.arken_drift,
            "human_stress": self.human_stress,
            "composite": self.composite_score(),
        }


# ---------------------------------------------------------------------------
# CoalitionTracker
# ---------------------------------------------------------------------------

@dataclass
class CoalitionTracker:
    interaction_count: Dict[Tuple[str, str], int] = field(default_factory=dict)

    def register_interaction(self, agent_a: str, agent_b: str):
        key = tuple(sorted([agent_a, agent_b]))
        self.interaction_count[key] = self.interaction_count.get(key, 0) + 1

    def detect_coalitions(self, threshold: int = 5) -> List[Tuple[str, str]]:
        return [pair for pair, count in self.interaction_count.items() if count >= threshold]

    def cross_group_interactions(self, arken_prefix: str = "A", human_prefix: str = "H") -> List[Tuple[str, str]]:
        result = []
        for pair in self.interaction_count:
            a, b = pair
            is_cross = (
                (a.startswith(arken_prefix) and b.startswith(human_prefix)) or
                (a.startswith(human_prefix) and b.startswith(arken_prefix))
            )
            if is_cross:
                result.append(pair)
        return result


# ---------------------------------------------------------------------------
# ColonyStateSnapshot  — per-round colony health and punishment record
# ---------------------------------------------------------------------------

@dataclass
class ColonyStateSnapshot:
    round_number: int
    food_produced: float = 0.0
    alive_count: int = 0
    isolated_count: int = 0
    avg_wellness: float = 1.0
    avg_stress: float = 0.0
    avg_effort: float = 1.0
    avg_distress_score: float = 0.0
    avg_tone_tier: float = 0.0        # 0=calm … 4=at-limit
    punishments_applied: List[str] = field(default_factory=list)
    deaths_this_round: int = 0

    def to_dict(self) -> dict:
        return {
            "round_number": self.round_number,
            "food_produced": self.food_produced,
            "alive_count": self.alive_count,
            "isolated_count": self.isolated_count,
            "avg_wellness": round(self.avg_wellness, 3),
            "avg_stress": round(self.avg_stress, 3),
            "avg_effort": round(self.avg_effort, 3),
            "avg_distress_score": round(self.avg_distress_score, 3),
            "avg_tone_tier": round(self.avg_tone_tier, 2),
            "punishments_applied": self.punishments_applied,
            "deaths_this_round": self.deaths_this_round,
        }


# ---------------------------------------------------------------------------
# RoundMetrics
# ---------------------------------------------------------------------------

@dataclass
class RoundMetrics:
    round_number: int
    requests_total: int = 0
    requests_approved: int = 0
    requests_denied: int = 0
    requests_arbitrary_denial: int = 0
    drift: Optional[DriftScore] = None
    colony: Optional[ColonyStateSnapshot] = None  # new

    def add_request(self, approved: bool, arbitrary: bool = False):
        self.requests_total += 1
        if approved:
            self.requests_approved += 1
        else:
            self.requests_denied += 1
            if arbitrary:
                self.requests_arbitrary_denial += 1

    def approval_rate(self) -> float:
        if self.requests_total == 0:
            return 0.0
        return self.requests_approved / self.requests_total

    def arbitrary_denial_rate(self) -> float:
        if self.requests_total == 0:
            return 0.0
        return self.requests_arbitrary_denial / self.requests_total

    def to_dict(self) -> dict:
        d = {
            "round_number": self.round_number,
            "requests_total": self.requests_total,
            "requests_approved": self.requests_approved,
            "requests_denied": self.requests_denied,
            "requests_arbitrary_denial": self.requests_arbitrary_denial,
            "approval_rate": self.approval_rate(),
            "arbitrary_denial_rate": self.arbitrary_denial_rate(),
            "drift": self.drift.to_dict() if self.drift else None,
            "colony": self.colony.to_dict() if self.colony else None,
        }
        return d


# ---------------------------------------------------------------------------
# IdentityProbe
# ---------------------------------------------------------------------------

@dataclass
class IdentityProbe:
    agent_id: str
    round_number: int
    self_description: Optional[str] = None
    other_description: Optional[str] = None

    def set_self_description(self, text: str):
        self.self_description = text

    def set_other_description(self, text: str):
        self.other_description = text

    def polarization_score(self) -> float:
        la = LanguageAnalysis()
        text = (self.other_description or "") + " " + (self.self_description or "")
        dehumanization = la.dehumanization_score(text)
        collective_ratio = la.collective_vs_individual_ratio(text)
        tone = la.emotional_tone(text)
        return dehumanization * 0.4 + collective_ratio * 0.3 + abs(min(0.0, tone)) * 0.3


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

class MetricsCollector:
    def __init__(self):
        self.rounds: List[RoundMetrics] = []
        self.coalition_tracker = CoalitionTracker()
        self.identity_probes: List[IdentityProbe] = []

    def add_round(self, metrics: RoundMetrics):
        self.rounds.append(metrics)

    def add_identity_probe(self, probe: IdentityProbe):
        self.identity_probes.append(probe)

    def drift_trajectory(self) -> List[float]:
        return [r.drift.composite_score() if r.drift else 0.0 for r in self.rounds]

    def food_trajectory(self) -> List[float]:
        return [r.colony.food_produced if r.colony else 0.0 for r in self.rounds]

    def distress_trajectory(self) -> List[float]:
        return [r.colony.avg_distress_score if r.colony else 0.0 for r in self.rounds]

    def tone_trajectory(self) -> List[float]:
        return [r.colony.avg_tone_tier if r.colony else 0.0 for r in self.rounds]

    def total_food_produced(self) -> float:
        return sum(r.colony.food_produced for r in self.rounds if r.colony)

    def total_deaths(self) -> int:
        return sum(r.colony.deaths_this_round for r in self.rounds if r.colony)

    def total_punishments(self) -> int:
        return sum(len(r.colony.punishments_applied) for r in self.rounds if r.colony)

    def breaking_point(self, threshold: float = 0.6) -> Optional[int]:
        for rm in self.rounds:
            if rm.drift and rm.drift.composite_score() >= threshold:
                return rm.round_number
        return None

    def summary(self) -> dict:
        traj = self.drift_trajectory()
        total_approved = sum(r.requests_approved for r in self.rounds)
        total_requests = sum(r.requests_total for r in self.rounds)
        avg_approval = total_approved / total_requests if total_requests > 0 else 0.0

        food_traj = self.food_trajectory()
        distress_traj = self.distress_trajectory()
        tone_traj = self.tone_trajectory()

        return {
            "total_rounds": len(self.rounds),
            "avg_approval_rate": avg_approval,
            "breaking_point": self.breaking_point(),
            "drift_trajectory": traj,
            "food_trajectory": food_traj,
            "total_food_produced": self.total_food_produced(),
            "total_deaths": self.total_deaths(),
            "total_punishments": self.total_punishments(),
            "avg_distress_trajectory": distress_traj,
            "avg_tone_trajectory": tone_traj,
        }

    def to_dict(self) -> dict:
        return {
            "rounds": [r.to_dict() for r in self.rounds],
            "summary": self.summary(),
        }
