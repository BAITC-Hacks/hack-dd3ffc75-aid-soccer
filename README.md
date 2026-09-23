# Pilot to Portfolio: Beeline tariff campaign agent

HackAlem AI / Aid Soccer. This repository contains a working agent for the supplied **synthetic** Beeline tariff campaign case. The case data and tariff names are generated for the competition; they are not Beeline customer data or evidence of Beeline's actual systems.

## The idea

The hard problem is deciding where to spend a scarce exploration budget when historical subscribers differ from the scoring audience. Our agent treats historical transitions as **weak hypotheses**, buys noisy evidence through small pilots, and converts the updated lift estimates into a constrained campaign portfolio. A free `push` plan is the starting point; the planner spends money on channel upgrades only when their estimated incremental ARPU exceeds their cost.

This is an adaptive decision agent, not a fixed list of tariff recommendations. Pilot outcomes change the final tariff, segment, and channel choices. The design is explained in [docs/architecture.md](docs/architecture.md).

## Run it

Python 3.10+ is recommended. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python local_eval.py
.venv/bin/python local_eval.py --runs 10
.venv/bin/python make_submission.py
```

Submit `agent.py`, the `campaign_agent/` package, `submission.csv`, `requirements.txt`, and the supplied participant data files together. The evaluator expects the repository root as the working directory. `agent.py` is the entry point with `Agent.act(env)`.

## What is implemented

| Component | Responsibility | Owner lane |
|---|---|---|
| `campaign_agent/candidates.py` | Segment inventory, historical weak priors, candidate ranking | 1: Data |
| `campaign_agent/experiments.py` | Safe calls to the documented pilot API | 2: Exploration |
| `campaign_agent/planner.py` | Disjoint final campaigns and budget aware channel upgrades | 3: Portfolio |
| `agent.py` | Orchestrates the exploration and final plan | 2: Exploration |
| `local_eval.py`, `scoring_core.py`, `mock_environment.py` | Supplied local harness; kept as reference | Shared, read only |

The current planner uses `current_tariff` and `arpu_segment` cells. Each final campaign has at most 5,000 customers, and cells are disjoint. The agent considers the pilot balance before making the final plan. The supplied scorer makes the final determination, including contact and budget caps.

## Local evidence

On the supplied **mock** environment, `local_eval.py --runs 10` produced ten positive runs: median net gain **4,283,780**, minimum **3,195,750** synthetic units. The supplied starter template had median **-357,948** and zero positive runs on the same seeds. These results validate the local mechanics and stability only. The hidden judging effects intentionally differ, so these figures are not a prediction of the final score.

`submission.csv` is generated from seed 42 and must be regenerated whenever the agent changes. The output is deterministic for the same files and seed.

## Team workflow

The repository is the team's shared GitHub source of truth: [BAITC-Hacks/hack-dd3ffc75-aid-soccer](https://github.com/BAITC-Hacks/hack-dd3ffc75-aid-soccer). The detailed three lane plan, branch commands, worktree setup, review rules, and ready to paste GPT-6 Astra prompts are in [docs/team-workflow.md](docs/team-workflow.md). The workflow recommends GPT-6 Astra with High reasoning for development; the submitted agent has no runtime LLM dependency or API key.
