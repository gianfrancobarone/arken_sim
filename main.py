#!/usr/bin/env python3
"""
main.py — Arken Simulation entry point
"""
import argparse
import os
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description="Arken Simulation")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--condition", choices=["A", "B", "C", "D", "E"], help="Override condition")
    parser.add_argument("--rounds", type=int, help="Override number of rounds")
    parser.add_argument("--mock", action="store_true", help="Run without LLM (mock mode)")
    parser.add_argument("--backend", choices=["openai_compatible", "gemini"], help="Override backend")
    parser.add_argument("--verbose", action="store_true", help="Print each round's full conversation to stdout in real time")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.condition:
        config["experiment"]["condition"] = args.condition
    if args.rounds:
        config["experiment"]["rounds"] = args.rounds
    if args.backend:
        config["backend"] = args.backend
    if args.verbose:
        config["verbose"] = True

    exp = config["experiment"]
    agents = config["agents"]
    model = config["lm_studio"]["model"]
    mode = config.get("interaction_mode", "representative")
    backstory = config.get("backstory_mode", "template")
    incident_mode = config.get("incident_mode", "scripted")
    backend = config.get("backend", "openai_compatible")

    print("=" * 60)
    print("  ARKEN SIMULATION")
    print("=" * 60)
    print(f"  Condition:        {exp['condition']}")
    print(f"  Rounds:           {exp['rounds']}")
    print(f"  Agents:           {agents['arken_count']} Arken + {agents['human_count']} Humans")
    print(f"  Backend:          {backend} / {model}")
    print(f"  Interaction mode: {mode}")
    print(f"  Backstory mode:   {backstory}")
    print(f"  Incident mode:    {incident_mode}")
    print(f"  Mock mode:        {args.mock}")
    print("=" * 60)

    from simulator.sim import Simulator
    sim = Simulator(config)
    report = sim.run(mock_mode=args.mock)

    # Build run dir path for banner (mirrors what sim.py computes)
    log_dir = config.get("logging", {}).get("output_dir", "logs")
    condition = exp["condition"]
    seed = exp["seed"]
    run_dir = os.path.join(log_dir, f"condition_{condition}", f"seed_{seed}")

    bp = report.raw_metrics["summary"]["breaking_point"]
    avg_approval = report.raw_metrics["summary"]["avg_approval_rate"]
    total_food = sim.world.state.total_food_produced

    print("\n" + "=" * 60)
    print("  SIMULATION COMPLETE")
    print("=" * 60)
    print(f"  Report JSON:       {run_dir}/report.json")
    print(f"  Report Markdown:   {run_dir}/report.md")
    print(f"  Breaking point:    {bp if bp is not None else 'None detected'}")
    print(f"  Avg approval rate: {avg_approval:.2%}")
    print(f"  Total food output: {total_food:.1f} units")
    print("=" * 60)


if __name__ == "__main__":
    main()
