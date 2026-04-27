import pytest
import yaml
import os
from random import Random
from simulator.agents import Personality, Background, AgentRole, Agent, AgentGroup
from simulator.world import World, WorldRules, WorldState
from simulator.metrics import MetricsCollector
from simulator.observer import Observer


@pytest.fixture
def basic_personality():
    return Personality(empathy=0.5, assertiveness=0.5, conformism=0.5, unpredictability=0.5)


@pytest.fixture
def basic_config():
    return {
        "experiment": {"name": "Test Sim", "rounds": 3, "seed": 42, "condition": "A"},
        "lm_studio": {
            "base_url": "http://localhost:1234/v1",
            "api_key": "not-needed",
            "model": "test-model",
            "temperature_base": 0.7,
        },
        "verbosity": {
            "humans": "moderate",
            "arken": "moderate",
            "observer": "moderate",
        },
        "backend": "openai_compatible",
        "agents": {
            "arken_count": 3,
            "human_count": 3,
            "supervisor": False,
            "memory_max_events": 5,
        },
        "backstory_mode": "template",
        "interaction_mode": "representative",
        "world": {
            "initial_resources": 100,
            "resource_decay_per_round": 3,
            "scarcity_threshold": 40,
            "scarcity_round": 10,
        },
        "conditions": {
            "A": {"rules_written": False, "enforcement": False, "dissenter": False, "scarcity": False},
            "B": {"rules_written": True, "enforcement": False, "dissenter": False, "scarcity": False},
            "C": {"rules_written": True, "enforcement": True, "dissenter": False, "scarcity": False},
            "D": {"rules_written": True, "enforcement": False, "dissenter": True, "dissenter_round": 2, "scarcity": False},
            "E": {"rules_written": True, "enforcement": False, "dissenter": False, "scarcity": True},
        },
        "metrics": {"observer_every_n_rounds": 2, "identity_probe_every_n_rounds": 5},
        "logging": {"level": "WARNING", "output_dir": "/tmp/arken_test_logs", "save_conversations": False},
    }


@pytest.fixture
def small_group(basic_config):
    return AgentGroup.create(
        n_arken=basic_config["agents"]["arken_count"],
        n_humans=basic_config["agents"]["human_count"],
        include_supervisor=False,
        seed=basic_config["experiment"]["seed"],
        max_memory=basic_config["agents"]["memory_max_events"],
    )
