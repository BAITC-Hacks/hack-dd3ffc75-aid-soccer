# Mock benchmark

Null means unavailable or nonfinite; failed runs are excluded from summary statistics.
Mock evaluator results do not predict judging rank; pilot gains are not added separately.
Final-only portfolio estimates cannot exactly deduplicate unknown pilot overlap.
Starter is evaluated unchanged, including pilot-only scores when no final campaigns are returned; see meets_final_count_requirement.

```json
{
  "adaptive": {
    "runs": 10,
    "successful_runs": 10,
    "failures": 0,
    "mean": 2898763.7671715785,
    "median": 3103097.097584981,
    "minimum": 1567730.5511855623,
    "standard_deviation": 756071.5308175391,
    "positive_runs": 10
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
    "absolute_improvement": 4554650.246943891
  },
  {
    "seed": 1,
    "absolute_improvement": 4239063.968308514
  },
  {
    "seed": 2,
    "absolute_improvement": 2746090.3359070285
  },
  {
    "seed": 3,
    "absolute_improvement": 2212802.605126644
  },
  {
    "seed": 4,
    "absolute_improvement": 4260070.474776736
  },
  {
    "seed": 5,
    "absolute_improvement": 4089552.923755061
  },
  {
    "seed": 6,
    "absolute_improvement": 2442404.764486607
  },
  {
    "seed": 7,
    "absolute_improvement": 3314292.0979584223
  },
  {
    "seed": 8,
    "absolute_improvement": 1644224.0031400716
  },
  {
    "seed": 9,
    "absolute_improvement": 4292848.089600488
  }
]
```

Mean absolute improvement: 3379599.9510003463
Median absolute improvement: 3461044.8939676243

## Failures
