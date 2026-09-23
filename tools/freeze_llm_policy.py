"""Install a successful live recommendation for deterministic offline replay."""
import argparse
import json
from pathlib import Path

from llm_advisor import AdviceUnavailable, freeze_advice


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'artifacts' / 'llm_policy.json')
    parser.add_argument('--replace', action='store_true', help='Explicitly replace an installed policy')
    args = parser.parse_args(argv)
    if args.output.exists() and not args.replace:
        parser.error('Policy already exists; use --replace only for a deliberate refresh')
    try:
        trace = json.loads(args.trace.read_text(encoding='utf-8'))
        policy = freeze_advice(trace['llm'])
    except (OSError, ValueError, KeyError, TypeError, AdviceUnavailable):
        parser.error('Input must be a trace with a successful live OpenAI recommendation')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(policy, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                           encoding='utf-8')
    print(f'Frozen policy saved to {args.output}; context={policy["context_hash"]}; API calls: 0')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
