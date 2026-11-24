"""Command line: ``bgtl train|transfer|eval --config FILE``."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bgtl")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "transfer", "eval"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict):
        raise SystemExit(f"{args.config} did not contain a mapping")
    if args.command == "train":
        from bgtl.train.loop import train

        train(config)
    elif args.command == "transfer":
        from bgtl.transfer.channel_map import transfer

        transfer(config)
    else:
        from bgtl.eval.match import evaluate

        evaluate(config)


if __name__ == "__main__":
    main()
