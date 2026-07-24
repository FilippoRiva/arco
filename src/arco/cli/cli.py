import argparse
import sys
import warnings

# Suppress all general UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress the specific LangChain warning
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    from arco.cli.commands import bench, run
    from arco.cli.console import console

    ## Parsing with argparse
    parser = argparse.ArgumentParser(
        description=("The arco cli to run or benchmark a workflow")
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
        sys.exit(1)


if __name__ == "__main__":
    main()
