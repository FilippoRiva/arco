import sys
from argparse import ArgumentParser, Namespace

from rich.pretty import pprint

from arco.cli import viz
from arco.cli.console import console
from arco.core import State


# ---------------------------------------------------------------------------
# Script Parser Registration
# ---------------------------------------------------------------------------
def register(subparsers: ArgumentParser) -> ArgumentParser:
    parser = subparsers.add_parser("cache", help="Invokes the cache management system to handle the local cache.")
    parser.add_argument("--save-dir", "-d", type=str, default="output", help="Directory of the cache to analyze")
    parser.add_argument("--clear", action="store_true", help="If set it entirely clears the cached runs")
    parser.add_argument("--delete", type=str, help="run_id of the run to delete")
    parser.add_argument("--runs", "-r", action="store_true", help="Shows the list of cached runs")
    parser.add_argument("--stats", "-s", action="store_true", help="Prints the cache statistics")
    parser.add_argument("--view-run", "-v", type=str, help="Visualize the specified cached run")
    return parser


# ---------------------------------------------------------------------------
# Script Handler
# ---------------------------------------------------------------------------
def handle(args: Namespace, parser: ArgumentParser) -> None:
    if not (args.clear or args.delete or args.runs or args.stats or args.view_run):  # defaults to help
        parser.print_help()
        sys.exit(1)

    # Load the dependencies
    with console.status("[bold cyan]Loading Cache...[/bold cyan]", spinner="dots"):
        from arco.data import RunCache

    # Run cache management
    save_dir = args.save_dir
    cache = RunCache(save_dir)
    if args.clear:
        if not confirm("Are you sure you want to clear ALL cache?"):
            return

        count = cache.clear_cache()
        console.print(f"[bold cyan]Cache cleared[/bold cyan] : {count} runs deleted")
        return
    elif args.delete:
        if not confirm(f"Delete run '{args.delete}'?"):
            return

        if cache.delete_run(args.delete):
            console.print(f"[bold cyan]Run deleted[/bold cyan] : run_id={args.delete}")
        else:
            console.print(f"[bold red]Failed to delete[/bold red] : run_id={args.delete} not found in cache")
        return

    if args.runs:
        runs = cache.list_runs()  # Assumed to be a list of dicts: [{}, {}, {}]
        console.print(f"[bold magenta]Available Runs[/bold magenta]({len(runs)}):")
        pprint(runs)
    if args.stats:
        stats = cache.get_cache_stats()  # Assumed to be a flat or nested dict
        console.print(f"[bold magenta]Cache Statistics[/bold magenta]:")
        pprint(stats)
    if args.view_run:
        metadata = cache.load_run_metadata(args.view_run)
        result: State = metadata['final_result']
        for answer in result.answers:
            console.print(viz.generate_answer_panel(answer, verbose=True))


def confirm(message: str) -> bool:
    ans = input(f"{message} [y/N]: ").strip().lower()
    return ans == "y"
