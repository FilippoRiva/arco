#!/usr/bin/env python3
import argparse
import sys
import warnings
from typing import TYPE_CHECKING

# Suppress all general UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress the specific LangChain warning
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    from arco.cli.commands import run, bench
    from arco.cli.console import console

    ## Parsing with argparse
    parser = argparse.ArgumentParser(
        description=(
            "The arco-cli utility tool to run the agent, manage cache or benchmark on ground-truth data"
        )
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    commands = {
        "run": run.register(subparsers),
        "benchmark": bench.register(subparsers),
    }

    handlers = {
        "run": run.handle,
        "benchmark": bench.handle,
    }

    # Parse
    args = parser.parse_args()

    # Run selected command
    try:
        if args.command in handlers:
            handlers[args.command](args, commands[args.command])
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        console.print("[bold red]Stopped[/bold red]: Keyboard Interrupt")
        exit(1)


if __name__ == "__main__":
    main()
