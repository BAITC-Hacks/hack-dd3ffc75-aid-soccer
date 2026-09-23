# Mock benchmark

**End-to-end evaluation: pending teammate commits.** Adaptive attempts below are
missing-dependency diagnostics, not completed performance measurements. See
`benchmark_integration_checklist.md` for the handoff and release steps.

Null means unavailable or nonfinite; failed runs are excluded from summary statistics.
Mock evaluator results do not predict judging rank; pilot gains are not added separately.
Final-only portfolio estimates cannot exactly deduplicate unknown pilot overlap.

```json
{
  "adaptive": {
    "runs": 10,
    "successful_runs": 0,
    "failures": 10,
    "mean": null,
    "median": null,
    "minimum": null,
    "standard_deviation": null,
    "positive_runs": 0
  },
  "starter": {
    "runs": 10,
    "successful_runs": 10,
    "failures": 0,
    "mean": -480836.1838287679,
    "median": -357947.7963826433,
    "minimum": -1019431.0606441048,
    "standard_deviation": 309734.3597929644,
    "positive_runs": 0
  }
}
```

## Absolute per-seed improvements

```json
[
  {
    "seed": 0,
    "absolute_improvement": null
  },
  {
    "seed": 1,
    "absolute_improvement": null
  },
  {
    "seed": 2,
    "absolute_improvement": null
  },
  {
    "seed": 3,
    "absolute_improvement": null
  },
  {
    "seed": 4,
    "absolute_improvement": null
  },
  {
    "seed": 5,
    "absolute_improvement": null
  },
  {
    "seed": 6,
    "absolute_improvement": null
  },
  {
    "seed": 7,
    "absolute_improvement": null
  },
  {
    "seed": 8,
    "absolute_improvement": null
  },
  {
    "seed": 9,
    "absolute_improvement": null
  }
]
```

Mean absolute improvement: None
Median absolute improvement: None

## Failures

- adaptive seed 0: ModuleNotFoundError: No module named 'candidate_engine'
- adaptive seed 1: ModuleNotFoundError: No module named 'candidate_engine'
- adaptive seed 2: ModuleNotFoundError: No module named 'candidate_engine'
- adaptive seed 3: ModuleNotFoundError: No module named 'candidate_engine'
- adaptive seed 4: ModuleNotFoundError: No module named 'candidate_engine'
- adaptive seed 5: ModuleNotFoundError: No module named 'candidate_engine'
- adaptive seed 6: ModuleNotFoundError: No module named 'candidate_engine'
- adaptive seed 7: ModuleNotFoundError: No module named 'candidate_engine'
- adaptive seed 8: ModuleNotFoundError: No module named 'candidate_engine'
- adaptive seed 9: ModuleNotFoundError: No module named 'candidate_engine'
