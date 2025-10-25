"""Convenience wrapper to run sed.infer from the CLI."""

from __future__ import annotations

from sed.infer import build_parser, run


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
