"""Per-execution logging setup with selective debug levels."""

import logging
from pathlib import Path


def initialize(run_id: str, log_dir: str | Path = "./logs", level: str = "INFO"):
    """Configure logging for a single workflow execution.

    Sets up a file handler that captures DEBUG+ for ``arco.*`` loggers
    and WARNING+ for all third-party libraries, keeping log files focused
    on your application's output.

    Args:
        run_id: Unique identifier for this run (used as the log filename).
        log_dir: Directory where log files are stored.
        level: Minimum log level for ``arco.*`` loggers.
            One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``.
            Third-party libraries always stay at ``WARNING`` or higher.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_id}.log"

    # Root logger: only WARNING+ by default (catches third-party libs)
    logging.getLogger().setLevel(logging.WARNING)

    # File handler captures everything
    handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(handler)

    # --- Your code: use the user-requested level ---
    level_value = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger("arco").setLevel(level_value)

    # --- Known noisy libraries: keep quiet ---
    for lib in (
        "langchain",
        "httpx",
        "openai",
        "urllib3",
        "duckdb",
        "matplotlib",
        "PIL",
    ):
        logging.getLogger(lib).setLevel(logging.WARNING)
    logging.getLogger("codecarbon").setLevel(logging.ERROR)
