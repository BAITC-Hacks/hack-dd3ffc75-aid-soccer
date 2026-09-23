"""Agent entry point for the supplied environment."""

from campaign_agent.candidates import _candidates, _prospects
from campaign_agent.experiments import _pilot
from campaign_agent.planner import _plan


class Agent:
    def act(self, env) -> list[dict]:
        candidates = _candidates(env)
        prospects = _prospects(candidates, min(14, env.pilots_left))
        tested = []
        for candidate in prospects:
            n = min(100, max(30, candidate.n // 3))
            if _pilot(env, candidate, n):
                tested.append(candidate)

        # Recheck leaders when uncertainty could materially change the plan.
        # Each second pilot is an independent sample of up to 200 customers.
        to_refine = sorted(
            tested,
            key=lambda c: c.n * c.arpu * max(0.0, c.posterior[0] + c.posterior[1]),
            reverse=True,
        )
        for candidate in to_refine[:min(6, env.pilots_left)]:
            if candidate.posterior[1] > 0.035:
                _pilot(env, candidate, min(200, candidate.n))
        return _plan(env, tested)
