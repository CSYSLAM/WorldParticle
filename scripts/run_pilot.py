"""Entry point for the first shared-model pilot; no implicit dependency installation."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["generate", "verify", "train", "evaluate"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pilot_v1.json")
    parser.add_argument("--family", choices=["softbody", "cloth", "joint", "rod"])
    args = parser.parse_args()
    if args.family and args.command != "generate":
        parser.error("--family is supported only by generate; evaluation runs all four families")
    config = json.loads(args.config.read_text())
    if args.command == "generate":
        from wp_vbd.generate import generate
        generate(config, only=args.family)
    elif args.command == "verify":
        from wp_vbd.verify import verify
        verify(config)
    elif args.command == "train":
        from wp_vbd.train import train
        train(config)
    else:
        from wp_vbd.evaluate import evaluate
        report = evaluate(config)
        print(f"Pilot inference finished: {report['status']}")
        print(f"Results and comparison animations: {ROOT / config['report_dir']}")
        if report["status"] != "PILOT_EXECUTED":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
