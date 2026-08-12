import argparse
import sys
import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

# Suppress all general UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress the specific LangChain warning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    from arco.cli.commands import analyze_benchmark, bench, generate_benchmark, run
    from arco.cli.console import console

    # Parsing with argparse
    parser = argparse.ArgumentParser(
        description=("The arco cli to run or benchmark a workflow")
    )

    # Add subcommands
    subparsers_action = parser.add_subparsers(
        dest="command", help="Available subcommands"
    )
    commands = {
        "run": run.register(subparsers_action),
        "generate-benchmark": generate_benchmark.register(subparsers_action),
        "benchmark": bench.register(subparsers_action),
        "analyze-benchmark": analyze_benchmark.register(subparsers_action),
    }

    handlers = {
        "run": run.handle,
        "generate-benchmark": generate_benchmark.handle,
        "benchmark": bench.handle,
        "analyze-benchmark": analyze_benchmark.handle,
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
        console.print("\n\n[bold red]Stopped[/bold red]: Keyboard Interrupt")
        sys.exit(1)


if __name__ == "__main__":
    main()
