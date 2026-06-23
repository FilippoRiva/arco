#!/usr/bin/env python3
import argparse
import os
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

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load the available commands
from arco.cli.commands import run, cache, bulk, bench, aggregate

# Load the console singleton
from arco.cli.console import console

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ## Parsing with argparse
    parser = argparse.ArgumentParser(
        description=(
            "The arco-cli utility tool to run the agent, manage cache, bulk execute for evaluation"
        )
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    commands = {
        "run": run.register(subparsers),
        "cache": cache.register(subparsers),
        "bulk": bulk.register(subparsers),
        "bench": bench.register(subparsers),
        "aggregate": aggregate.register(subparsers),
    }

    handlers = {
        "run": run.handle,
        "cache": cache.handle,
        "bulk": bulk.handle,
        "bench": bench.handle,
        "aggregate": aggregate.handle,
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


if __name__ == "__main__":
    main()
